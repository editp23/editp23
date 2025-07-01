import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import DDPMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from pipeline import Zero123PlusPipeline
from utils import add_white_bg, load_z123_pipe
from typing import Optional

class VAEProcessor:
    """A helper class to handle encoding and decoding images with the VAE."""
    def __init__(self, pipeline: Zero123PlusPipeline):
        self.pipe = pipeline
        self.image_processor = pipeline.image_processor
        self.vae = pipeline.vae
        
        self.latent_shift_factor = 0.22
        self.latent_scale_factor = 0.75
        self.image_scale_factor = 0.5 / 0.8

    def encode(self, image: Image.Image) -> torch.Tensor:
        """Encodes a PIL image into the latent space."""
        image_tensor = self.image_processor.preprocess(image).to(self.vae.device).half()
        with torch.autocast("cuda"), torch.inference_mode():
            image_tensor *= self.image_scale_factor
            denorm = self.vae.encode(image_tensor).latent_dist.mode()
            denorm *= self.vae.config.scaling_factor
        return (denorm - self.latent_shift_factor) * self.latent_scale_factor

    def decode(self, latents: torch.Tensor) -> Image.Image:
        """Decodes latents back into a post-processed image."""
        with torch.autocast("cuda"), torch.inference_mode():
            denorm = latents / self.latent_scale_factor + self.latent_shift_factor
            image = self.vae.decode(denorm / self.vae.config.scaling_factor, return_dict=False)[0]
            image /= self.image_scale_factor
        return self.image_processor.postprocess(image)


class EditAwareDenoiser:
    """Encapsulates the entire Edit-Aware Denoising process."""
    def __init__(self, pipe: Zero123PlusPipeline, scheduler: DDPMScheduler, T_steps: int, src_gs: float, tar_gs: float, n_max: int):
        """Initializes the denoiser with the pipeline and configuration."""
        self.pipe = pipe
        self.scheduler = scheduler
        self.T_steps = T_steps
        self.src_guidance_scale = src_gs
        self.tar_guidance_scale = tar_gs
        self.n_max = n_max

    @staticmethod
    def _mix_cfg(cond: torch.Tensor, uncond: torch.Tensor, cfg: float) -> torch.Tensor:
        """Mixes conditional and unconditional predictions."""
        return uncond + cfg * (cond - uncond)

    def _get_differential_edit_direction(self, t: torch.Tensor, zt_src: torch.Tensor, zt_tar: torch.Tensor) -> torch.Tensor:
        """Computes the differential edit direction (delta v) for a timestep."""
        condition_noise = torch.randn_like(self.src_cond_lat)

        noisy_src_cond_lat = self.pipe.scheduler.scale_model_input(
            self.pipe.scheduler.add_noise(self.src_cond_lat, condition_noise, t), t
        )
        vt_src_uncond, vt_src_cond = self._calc_v_zero(self.src_cond_img, zt_src, t, noisy_src_cond_lat)
        vt_src = self._mix_cfg(vt_src_cond, vt_src_uncond, self.src_guidance_scale)

        noisy_tar_cond_lat = self.pipe.scheduler.scale_model_input(
            self.pipe.scheduler.add_noise(self.tar_cond_lat, condition_noise, t), t
        )
        vt_tar_uncond, vt_tar_cond = self._calc_v_zero(self.tar_cond_img, zt_tar, t, noisy_tar_cond_lat)
        vt_tar = self._mix_cfg(vt_tar_cond, vt_tar_uncond, self.tar_guidance_scale)

        return vt_tar - vt_src

    def _propagate_for_timestep(self, zt_edit: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Performs a single propagation step for the edit."""
        fwd_noise = torch.randn_like(self.x_src)
        zt_src = self.scheduler.scale_model_input(self.scheduler.add_noise(self.x_src, fwd_noise, t), t)
        zt_tar = self.scheduler.scale_model_input(self.scheduler.add_noise(zt_edit, fwd_noise, t), t)

        diff_v = self._get_differential_edit_direction(t, zt_src, zt_tar)
        
        zt_edit_change = dt * diff_v
        zt_edit = zt_edit.to(torch.float32) + zt_edit_change
        return zt_edit.to(diff_v.dtype)

    def _calc_v_zero(self, condition_image: Image.Image, noisy_latent: torch.Tensor, t: torch.Tensor, noised_condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the unconditional and conditional v-prediction from the UNet."""
        DUMMY_GUIDANCE_SCALE = 2
        model_output = {}

        def hook_fn(module, args, output):
            model_output['v_pred'] = output[0]

        hook_handle = self.pipe.unet.register_forward_hook(hook_fn)

        try:
            self.pipe(
                condition_image,
                latents=noisy_latent,
                num_inference_steps=1,
                guidance_scale=DUMMY_GUIDANCE_SCALE,
                timesteps=[t.item()],
                output_type="latent",
                noisy_cond_lat=noised_condition,
            )
        finally:
            hook_handle.remove()

        return model_output['v_pred'].chunk(2)

    @torch.no_grad()
    def denoise(self, x_src: torch.Tensor, src_cond_img: Image.Image, tar_cond_img: Image.Image) -> torch.Tensor:
        """Public method to run the entire denoising process."""
        self.x_src = x_src
        self.src_cond_img = src_cond_img
        self.tar_cond_img = tar_cond_img
        
        timesteps, _ = retrieve_timesteps(self.scheduler, self.T_steps, self.x_src.device)
        zt_edit = self.x_src.clone()

        self.src_cond_lat = self.pipe.make_condition_lat(self.src_cond_img, guidance_scale=2.0)
        self.tar_cond_lat = self.pipe.make_condition_lat(self.tar_cond_img, guidance_scale=2.0)
        
        start_index = max(0, len(timesteps) - self.n_max)

        for i in tqdm(range(start_index, len(timesteps))):
            t = timesteps[i]
            t_i = t / 1000.0
            t_im1 = timesteps[i + 1] / 1000.0 if i + 1 < len(timesteps) else torch.zeros_like(t_i)
            dt = t_im1 - t_i
            
            zt_edit = self._propagate_for_timestep(zt_edit, t, dt)

        return zt_edit


def run_editp23(
    src_condition_path: str,
    tgt_condition_path: str,
    original_mv: str,
    save_path: str,
    device_number: int = 0,
    T_steps: int = 50,
    n_max: int = 31,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 5.0,
    seed: int = 18,
    pipeline: Optional[Zero123PlusPipeline] = None,
) -> None:
    """Main execution function to run the complete editing pipeline."""
    if pipeline is None:
        pipeline = load_z123_pipe(device_number)
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    vae_processor = VAEProcessor(pipeline)

    src_cond_img = add_white_bg(Image.open(src_condition_path))
    tgt_cond_img = add_white_bg(Image.open(tgt_condition_path))
    mv_src = add_white_bg(Image.open(original_mv))
    x0_src = vae_processor.encode(mv_src)

    denoiser = EditAwareDenoiser(
        pipe=pipeline,
        scheduler=pipeline.scheduler,
        T_steps=T_steps,
        src_gs=src_guidance_scale,
        tar_gs=tar_guidance_scale,
        n_max=n_max
    )
    x0_tar = denoiser.denoise(x0_src, src_cond_img, tgt_cond_img)

    image_tar = vae_processor.decode(x0_tar)
    image_tar[0].save(save_path)
    print(f"Successfully saved result to {save_path}")