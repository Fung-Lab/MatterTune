from __future__ import annotations

import argparse
from pathlib import Path

from mattersim.lammps.mattertune_mliap_wrapper import MatterTuneMatterSimMLIAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a MatterTune-finetuned MatterSim checkpoint for LAMMPS ML-IAP."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="MatterTune Lightning checkpoint produced by MatterSim fine-tuning.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output .pt file for LAMMPS pair_style mliap unified.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used while loading the checkpoint. The exported model is saved on CPU.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict Lightning state_dict loading. Default is non-strict for MatterSim version compatibility.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile at LAMMPS runtime.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mliap = MatterTuneMatterSimMLIAP.from_checkpoint(
        args.checkpoint,
        device=args.device,
        strict=args.strict,
        compile=not args.no_compile,
    )
    mliap.save(str(args.output))

    has_denormalizer = mliap.energy_denormalizer is not None
    print(f"wrote {args.output}")
    print(f"energy_denormalizer={has_denormalizer}")
    print(f"cutoff={mliap.cutoff} threebody_cutoff={mliap.threebody_cutoff}")


if __name__ == "__main__":
    main()
