import torch
import os
from diffusers import DDPMScheduler
from pipeline import Zero123PlusPipeline
from huggingface_hub import hf_hub_download
from PIL import Image


def load_z123_pipe(device_number):
    device = torch.device(
        f"cuda:{device_number}" if torch.cuda.is_available() else "cpu"
    )

    pipeline = Zero123PlusPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", torch_dtype=torch.float16
    )
    # DDPM supports custom timesteps
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    unet_path = "ckpts/diffusion_pytorch_model.bin"
    # load custom white-background UNet
    if os.path.exists(unet_path):
        unet_ckpt_path = unet_path
    else:
        unet_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="diffusion_pytorch_model.bin",
            repo_type="model",
        )
    state_dict = torch.load(unet_ckpt_path, map_location="cpu")
    pipeline.unet.load_state_dict(state_dict, strict=True)

    pipeline.to(device)
    return pipeline


def add_white_bg(image):
    # Check if image has transparency (RGBA or LA mode)
    if image.mode in ("RGBA", "LA"):
        # Create a white background image of the same size
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        # Paste original image onto white background using alpha channel as mask
        white_bg.paste(image, mask=image.split()[-1])
        return white_bg
    # If no transparency, return the original image
    return image
