import argparse
import sys
from pathlib import Path
from edit_mv import run_editp23, load_z123_pipe

def main(args: argparse.Namespace) -> None:
    """
    Sets up and runs the EditP23 process for a single experiment.
    """
    exp_dir = Path(args.exp_dir)
    input_files = {
        "src_path": exp_dir / "src.png",
        "edited_path": exp_dir / "edited.png",
        "src_mv_path": exp_dir / "src_mv.png",
    }

    # Pre-run validation to ensure all input files exist
    for name, path in input_files.items():
        if not path.is_file():
            print(f"Error: Input file not found at {path}")
            sys.exit(1)

    output_dir = exp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / f"result_tgs_{args.tar_guidance_scale}_nmax_{args.n_max}.png"

    print(f"Running edit for experiment: {args.exp_dir}")
    print(f"Saving to: {save_path}")

    pipeline = load_z123_pipe(args.device_number)

    run_editp23(
        src_condition_path=str(input_files["src_path"]),
        tgt_condition_path=str(input_files["edited_path"]),
        original_mv=str(input_files["src_mv_path"]),
        save_path=str(save_path),
        device_number=args.device_number,
        T_steps=args.T_steps,
        n_max=args.n_max,
        src_guidance_scale=args.src_guidance_scale,
        tar_guidance_scale=args.tar_guidance_scale,
        seed=args.seed,
        pipeline=pipeline,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Run EditP23 for 3D object editing.
Paper presets for (tar_guidance_scale, n_max):
- Mild: (5, 31)
- Medium: (6, 41), (12, 42)
- Hard: (21, 39)""",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to the experiment directory. Expects src.png, edited.png, and src_mv.png in this directory.")
    parser.add_argument("--device_number", type=int, default=0,
                        help="GPU device number to use.")
    parser.add_argument("--seed", type=int, default=18,
                        help="Random seed for reproducibility.")
    parser.add_argument("--T_steps", type=int, default=50,
                        help="Total number of denoising steps.")
    parser.add_argument("--n_max", type=int, default=31,
                        help="Number of scheduler steps for edit-aware guidance. Increase up to T_steps for more significant edits.")
    parser.add_argument("--src_guidance_scale", type=float, default=3.5,
                        help="CFG scale for the source condition. Can typically remain constant.")
    parser.add_argument("--tar_guidance_scale", type=float, default=5.0,
                        help="CFG scale for the target condition. Increase for more significant edits.")

    args = parser.parse_args()
    main(args)