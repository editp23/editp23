import argparse
import sys
from pathlib import Path
from typing import Optional

# --- Start of the "Messy" but Effective Path Setup ---
# This block ensures that imports work correctly without modifying the src directory.
# It adds both the project root and the src directory to the Python path.
try:
    # Get the project root directory (which is the parent of the 'scripts' directory)
    project_root = Path(__file__).resolve().parent.parent
    # Get the source code directory
    src_dir = project_root / "src"

    # Add both directories to the system path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_dir))
except IndexError:
    # Fallback for when the script is run in a way that __file__ is not defined
    print("Could not determine project root. Please run from the 'scripts' directory.")
    sys.exit(1)
# --- End of Path Setup ---

import torch
from PIL import Image

from pipeline import Zero123PlusPipeline # This now works because src/ is on the path
from utils import add_white_bg, load_z123_pipe


def generate_from_single_view(
    input_path: Path,
    output_path: Path,
    device_number: int = 0,
    pipeline: Optional[Zero123PlusPipeline] = None,
) -> None:
    """
    Generates a multi-view image grid from a single input image.

    Args:
        input_path: Path to the single input image.
        output_path: Path to save the generated multi-view .png file.
        device_number: The GPU device number to use.
        pipeline: An optional pre-loaded pipeline instance.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found at: {input_path}")

    print(f"Loading pipeline on device {device_number}...")
    if pipeline is None:
        pipeline = load_z123_pipe(device_number)

    print(f"Processing input image: {input_path}")
    cond_image = Image.open(input_path)
    cond_image = add_white_bg(cond_image)

    print("Generating multi-view grid...")
    result = pipeline(cond_image, num_inference_steps=75).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Successfully saved multi-view grid to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a multi-view image grid from a single input view using Zero123++."
    )
    parser.add_argument(
        "--input_image",
        type=Path,
        required=True,
        help="Path to the single input image file (e.g., examples/robot_sunglasses/src.png)."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the output multi-view grid (e.g., examples/robot_sunglasses/src_mv.png)."
    )
    parser.add_argument(
        "--device_number",
        type=int,
        default=0,
        help="GPU device number to use for generation."
    )
    args = parser.parse_args()

    try:
        generate_from_single_view(
            input_path=args.input_image,
            output_path=args.output_path,
            device_number=args.device_number
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
