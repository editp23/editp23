import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
submodule_path = os.path.join(script_dir, "..", "external", "instant-mesh")
sys.path.insert(0, submodule_path)

from src.utils.camera_util import (
    get_circular_camera_poses,
    get_zero123plus_input_cameras,
    FOV_to_intrinsics,
)
from src.utils.train_util import instantiate_from_config
from src.utils.mesh_util import save_obj
from src.utils.infer_util import save_video


def get_render_cameras(
    batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False
):
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = (
            FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        )
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(
    model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False
):
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes, render_cameras[:, i : i + chunk_size], render_size=render_size
            )["img"]
        else:
            frame = model.forward_synthesizer(
                planes, render_cameras[:, i : i + chunk_size], render_size=render_size
            )["images_rgb"]
        frames.append(frame)
    frames = torch.cat(frames, dim=1)[0]
    return frames

def main(args):
    """
    Main function to run the 3D mesh generation process.
    """
    # ============================
    # CONFIG
    # ============================
    print("🚀 Starting 3D mesh generation...")
    config = OmegaConf.load(args.config)
    config_name = os.path.basename(args.config).replace(".yaml", "")
    model_config = config.model_config
    infer_config = config.infer_config
    IS_FLEXICUBES = config_name.startswith("instant-mesh")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================
    # SETUP OUTPUT DIRECTORY
    # ============================
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    mesh_path = os.path.join(args.output_dir, "recon.obj")
    video_path = os.path.join(args.output_dir, "recon.mp4")

    # ============================
    # LOAD RECONSTRUCTION MODEL
    # ============================
    print("Loading reconstruction model...")
    model = instantiate_from_config(model_config)

    # Download model checkpoint if it doesn't exist
    model_ckpt_path = (
        infer_config.model_path
        if os.path.exists(infer_config.model_path)
        else hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename=f"{config_name.replace('-', '_')}.ckpt",
            repo_type="model",
        )
    )

    # Load the state dictionary
    state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {
        k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")
    }
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)

    # ============================
    # PREPARE DATA
    # ============================
    print(f"Processing input file: {args.input_file}")
    
    # Load and preprocess the input image
    input_image = Image.open(args.input_file).convert("RGB")
    images = np.asarray(input_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    # Rearrange from (C, H, W) to (B, C, H, W) where B is the number of views
    images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)
    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, size=320, interpolation=3, antialias=True).clamp(0, 1)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * args.scale).to(device)

    # ============================
    # RUN INFERENCE AND SAVE OUTPUT
    # ============================
    with torch.no_grad():
        # Generate 3D mesh
        planes = model.forward_planes(images, input_cameras)
        mesh_out = model.extract_mesh(planes, use_texture_map=False, **infer_config)
        
        # Save the mesh
        vertices, faces, vertex_colors = mesh_out
        save_obj(vertices, faces, vertex_colors, mesh_path)
        print(f"✅ Mesh saved to {mesh_path}")

        # Render and save video if enabled
        if args.save_video:
            print("🎥 Rendering video...")
            render_size = infer_config.render_resolution
            chunk_size = 20 if IS_FLEXICUBES else 1
            render_cameras = get_render_cameras(
                batch_size=1,
                M=120,
                radius=args.distance,
                elevation=20.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)

            frames = render_frames(
                model=model,
                planes=planes,
                render_cameras=render_cameras,
                render_size=render_size,
                chunk_size=chunk_size,
                is_flexicubes=IS_FLEXICUBES,
            )
            save_video(frames, video_path, fps=30)
            print(f"✅ Video saved to {video_path}")

    print("✨ Process complete.")

if __name__ == "__main__":
    # ============================
    # SCRIPT ARGUMENTS
    # ============================
    parser = argparse.ArgumentParser(
        description="Generate a 3D mesh and video from a single multi-view PNG file."
    )
    
    # Positional argument for config file
    parser.add_argument(
        "config", 
        type=str, 
        help="Path to the model config file (.yaml)."
    )
    
    # Required file paths
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input PNG file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/", 
        help="Directory to save the output .obj and .mp4 files. Defaults to 'outputs/'."
    )
    
    # Optional parameters for model and rendering
    parser.add_argument(
        "--scale", 
        type=float, 
        default=1.0, 
        help="Scale of the input cameras."
    )
    parser.add_argument(
        "--distance", 
        type=float, 
        default=4.5, 
        help="Camera distance for rendering the output video."
    )
    parser.add_argument(
        "--no_video", 
        dest="save_video", 
        action="store_false", 
        help="If set, disables saving the output .mp4 video."
    )
    
    parsed_args = parser.parse_args()
    main(parsed_args)
