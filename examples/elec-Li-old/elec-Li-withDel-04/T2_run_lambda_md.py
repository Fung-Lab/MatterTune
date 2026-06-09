from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/electrolyte")
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "local_runs" / "elec-Li-withDel-04"
DEFAULT_CONFIG_TYPE = "case1-case3-Li-FSI-FEC-1-13.0"
DEFAULT_STRUCTURE = Path(
    "/storage/lingyu/Electrolyte/get_all_trj_aimd/Li-metal/case3-Li-FSI-FEC/"
    "case1-case3-Li-FSI-FEC-1-13.0/top.pdb"
)
DEFAULT_MD_ROOT = Path("/storage/lingyu/Electrolyte/MLIP-MD")


def find_latest_checkpoint() -> Path:
    candidates = sorted(
        DEFAULT_OUTPUT_ROOT.glob("*/checkpoints/*best.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No best checkpoint found under {DEFAULT_OUTPUT_ROOT}. "
            "Pass --checkpoint explicitly."
        )
    return candidates[-1]


def lambda_label(value: float) -> str:
    return f"{value:.6g}"


def default_output_dir(config_type: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_MD_ROOT / config_type / stamp


def build_command(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> list[str]:
    lam = lambda_label(args.lambda_value)
    cmd = [
        sys.executable,
        "examples/electrolyte/md.py",
        "--ckpt-path",
        str(checkpoint),
        "--device",
        args.device,
        "--steps",
        str(args.steps),
        "--structure",
        str(args.structure),
        "--target-indices",
        args.target_indices,
        "--lambda-value",
        str(args.lambda_value),
        "--temperature",
        str(args.temperature),
        "--thermostat",
        args.thermostat,
        "--thermostat-timecon-fs",
        str(args.thermostat_timecon_fs),
        "--timestep-fs",
        str(args.timestep_fs),
        "--friction-fs-inv",
        str(args.friction_fs_inv),
        "--log-interval",
        str(args.log_interval),
        "--trajectory-interval",
        str(args.trajectory_interval),
        "--diagnostics-interval",
        str(args.diagnostics_interval),
        "--sigma",
        str(args.sigma),
        "--epsilon",
        str(args.epsilon),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
        "--trajectory-name",
        f"md_lambda_{lam}.xyz",
        "--energy-log-name",
        f"energy_lambda_{lam}.csv",
        "--final-structure-name",
        f"final_lambda_{lam}.extxyz",
        "--diagnostics-name",
        f"diagnostics_lambda_{lam}.jsonl",
    ]
    if args.init_velocities:
        cmd.append("--init-velocities")
    if args.use_d3:
        cmd.append("--use-d3")
    if args.no_smooth:
        cmd.append("--no-smooth")
    return cmd


def write_info(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one lambda-MD test with a specified checkpoint and structure."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--config-type", default=DEFAULT_CONFIG_TYPE)
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--target-indices", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--thermostat", choices=("langevin", "bussi"), default="bussi")
    parser.add_argument("--thermostat-timecon-fs", type=float, default=100.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--friction-fs-inv", type=float, default=0.02)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--trajectory-interval", type=int, default=100)
    parser.add_argument("--diagnostics-interval", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=2.337)
    parser.add_argument("--epsilon", type=float, default=0.00694)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-velocities", dest="init_velocities", action="store_true", default=False)
    parser.add_argument("--no-init-velocities", dest="init_velocities", action="store_false")
    parser.add_argument("--use-d3", action="store_true")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be non-negative.")
    if args.thermostat_timecon_fs <= 0.0:
        parser.error("--thermostat-timecon-fs must be positive.")
    for name in ("log_interval", "trajectory_interval", "diagnostics_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 0.0 <= args.lambda_value <= 1.0:
        parser.error("--lambda-value must be in [0, 1].")
    if not args.structure.is_file():
        raise FileNotFoundError(args.structure)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    return args


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or find_latest_checkpoint()
    output_dir = args.out_dir or default_output_dir(args.config_type)
    cmd = build_command(args, checkpoint, output_dir)
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "checkpoint": str(checkpoint),
        "structure": str(args.structure),
        "config_type": args.config_type,
        "lambda_value": args.lambda_value,
        "target_indices": args.target_indices,
        "output_dir": str(output_dir),
        "command": cmd,
        "parameters": json_ready(
            vars(args) | {"checkpoint": str(checkpoint), "out_dir": str(output_dir)}
        ),
    }
    write_info(output_dir / "T2_info.json", info)
    print(f"Wrote {output_dir / 'T2_info.json'}")
    print(" ".join(cmd))
    if args.dry_run:
        return

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
