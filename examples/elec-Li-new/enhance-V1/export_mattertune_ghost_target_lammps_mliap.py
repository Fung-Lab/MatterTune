from __future__ import annotations

import argparse
from pathlib import Path

from mattersim.lammps.ghost_target_mliap_wrapper import GhostTargetMatterSimMLIAP


def parse_target_types(value: str) -> tuple[int, ...]:
    target_types = tuple(
        int(piece.strip())
        for piece in value.split(",")
        if piece.strip()
    )
    if not target_types:
        raise argparse.ArgumentTypeError("--target-types must not be empty.")
    if any(atom_type <= 0 for atom_type in target_types):
        raise argparse.ArgumentTypeError("LAMMPS atom types must be positive.")
    return target_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a MatterTune-finetuned MatterSim checkpoint as a ghost-target "
            "LAMMPS ML-IAP model."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--target-types", type=parse_target_types, required=True)
    parser.add_argument("--epsilon", type=float, default=0.00694)
    parser.add_argument("--sigma", type=float, default=2.337)
    parser.add_argument(
        "--lj-cutoff",
        type=float,
        default=10.0,
        help=(
            "Requested LJ cutoff in Angstrom. At LAMMPS runtime the wrapper "
            "uses min(lj_cutoff, Lmin/2) and counts only the nearest image "
            "for each target-environment atom pair."
        ),
    )
    parser.add_argument(
        "--ghost-checkpoint",
        type=Path,
        default=None,
        help="Optional separate MatterTune MatterSim checkpoint for the ghost endpoint.",
    )
    parser.add_argument(
        "--energy-log-path",
        type=Path,
        default=None,
        help=(
            "Optional CSV path written by the LAMMPS ML-IAP wrapper at runtime. "
            "The file is overwritten when LAMMPS first evaluates forces."
        ),
    )
    parser.add_argument(
        "--energy-log-interval",
        type=int,
        default=1,
        help="Write one FEP-TI energy CSV row every N ML-IAP force evaluations.",
    )
    parser.add_argument(
        "--energy-log-timestep-fs",
        type=float,
        default=1.0,
        help="Timestep in fs used to populate time_fs/time_ps in the energy CSV.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used while loading checkpoint(s). The exported model is saved on CPU.",
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
    args = parser.parse_args()

    if not 0.0 <= args.lambda_value <= 1.0:
        parser.error("--lambda-value must be in [0, 1].")
    if args.sigma <= 0.0:
        parser.error("--sigma must be positive.")
    if args.lj_cutoff <= 0.0:
        parser.error("--lj-cutoff must be positive.")
    if args.energy_log_interval <= 0:
        parser.error("--energy-log-interval must be positive.")
    if args.energy_log_timestep_fs <= 0.0:
        parser.error("--energy-log-timestep-fs must be positive.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.ghost_checkpoint is not None and not args.ghost_checkpoint.is_file():
        raise FileNotFoundError(args.ghost_checkpoint)
    return args


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mliap = GhostTargetMatterSimMLIAP.from_mattertune_checkpoint(
        args.checkpoint,
        ghost_checkpoint=args.ghost_checkpoint,
        lambda_value=args.lambda_value,
        target_types=args.target_types,
        epsilon=args.epsilon,
        sigma=args.sigma,
        lj_cutoff=args.lj_cutoff,
        energy_log_path=args.energy_log_path,
        energy_log_interval=args.energy_log_interval,
        energy_log_timestep_fs=args.energy_log_timestep_fs,
        device=args.device,
        strict=args.strict,
        compile=not args.no_compile,
    )
    mliap.save(str(args.output))

    print(f"wrote {args.output}")
    print(f"lambda_value={mliap.lambda_value}")
    print(f"target_types={','.join(str(t) for t in mliap.target_types)}")
    print(f"epsilon={mliap.epsilon} sigma={mliap.sigma} lj_cutoff={mliap.lj_cutoff}")
    if mliap.energy_log_path is not None:
        print(
            f"energy_log_path={mliap.energy_log_path} "
            f"energy_log_interval={mliap.energy_log_interval} "
            f"energy_log_timestep_fs={mliap.energy_log_timestep_fs}"
        )
    print(f"cutoff={mliap.cutoff} threebody_cutoff={mliap.threebody_cutoff}")
    print(
        f"ghost_cutoff={mliap.ghost_cutoff} "
        f"ghost_threebody_cutoff={mliap.ghost_threebody_cutoff}"
    )


if __name__ == "__main__":
    main()
