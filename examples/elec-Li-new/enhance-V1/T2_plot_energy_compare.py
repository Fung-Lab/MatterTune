from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import iread


HARTREE_TO_EV = 27.211386
DEFAULT_AIMD_DAT = (
    Path(__file__).resolve().parents[2]
    / "electrolyte"
    / "AIMD_results"
    / "case3-Li-FSI-FEC_case1-case3-Li-FSI-FEC-1-13.0_lambda_0.50.dat"
)


def filter_max_time(
    data: dict[str, np.ndarray],
    *,
    max_time_ps: float | None,
) -> dict[str, np.ndarray]:
    if max_time_ps is None:
        return data
    mask = data["time_ps"] <= max_time_ps + 1.0e-12
    if not np.any(mask):
        raise ValueError(f"No samples remain after applying max_time_ps={max_time_ps}.")
    return {key: value[mask] for key, value in data.items()}


def load_aimd_dat(
    path: Path,
    *,
    last_n_frames: int | None,
    max_time_ps: float | None,
) -> dict[str, np.ndarray]:
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] < 5:
        raise ValueError(f"Expected at least 5 columns in {path}, got shape {raw.shape}.")
    if last_n_frames is not None and last_n_frames > 0:
        raw = raw[-last_n_frames:]

    time_ps = raw[:, 1].astype(float) / 1000.0
    time_ps = time_ps - time_ps[0]
    e_i = raw[:, 3].astype(float) * HARTREE_TO_EV
    e_f = raw[:, 4].astype(float) * HARTREE_TO_EV
    data = {
        "time_ps": time_ps,
        "frame": np.arange(raw.shape[0], dtype=float),
        "E_I_eV": e_i,
        "E_F_eV": e_f,
        "deltaE_eV": e_f - e_i,
    }
    return filter_max_time(data, max_time_ps=max_time_ps)


def load_mlip_energy_log(path: Path, *, mlip_ef_mode: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    e_f_column = "E_F_without_LJ_eV" if mlip_ef_mode == "without_lj" else "E_F_with_LJ_eV"
    delta_column = "deltaE_without_LJ_eV" if mlip_ef_mode == "without_lj" else "deltaE_with_LJ_eV"
    for column in ("time_ps", "E_I_eV", e_f_column, delta_column):
        if column not in df.columns:
            raise ValueError(f"{path} missing required column {column!r}.")
    return {
        "time_ps": df["time_ps"].to_numpy(dtype=float),
        "frame": df["step"].to_numpy(dtype=float) if "step" in df.columns else np.arange(len(df)),
        "E_I_eV": df["E_I_eV"].to_numpy(dtype=float),
        "E_F_eV": df[e_f_column].to_numpy(dtype=float),
        "deltaE_eV": df[delta_column].to_numpy(dtype=float),
    }


def load_mlip_diagnostics(path: Path, *, mlip_ef_mode: str) -> dict[str, np.ndarray]:
    time_fs: list[float] = []
    e_i: list[float] = []
    e_f: list[float] = []
    delta: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            real = record["real_endpoint"]
            ghost = record["ghost_endpoint"]
            time_fs.append(float(record["time_fs"]))
            e_i_value = float(real["base_energy_eV"])
            e_f_value = (
                float(ghost["base_energy_eV"])
                if mlip_ef_mode == "without_lj"
                else float(ghost["total_energy_eV"])
            )
            e_i.append(e_i_value)
            e_f.append(e_f_value)
            delta.append(e_f_value - e_i_value)
    time_ps = np.asarray(time_fs, dtype=float) / 1000.0
    return {
        "time_ps": time_ps,
        "frame": np.arange(time_ps.size, dtype=float),
        "E_I_eV": np.asarray(e_i, dtype=float),
        "E_F_eV": np.asarray(e_f, dtype=float),
        "deltaE_eV": np.asarray(delta, dtype=float),
    }


def load_mlip_xyz(path: Path, *, mlip_ef_mode: str, max_frames: int | None) -> dict[str, np.ndarray]:
    time_ps: list[float] = []
    e_i: list[float] = []
    e_f: list[float] = []
    delta: list[float] = []
    for frame_index, atoms in enumerate(iread(path, ":")):
        if max_frames is not None and frame_index >= max_frames:
            break
        info = atoms.info
        e_i_value = float(info["lambda0_energy_eV"])
        if mlip_ef_mode == "without_lj":
            e_f_value = float(info["lambda1_base_energy_eV"])
        else:
            e_f_value = float(info["lambda1_energy_eV"])
        time_ps.append(float(info.get("time_fs", frame_index)) / 1000.0)
        e_i.append(e_i_value)
        e_f.append(e_f_value)
        delta.append(e_f_value - e_i_value)
    time_ps_array = np.asarray(time_ps, dtype=float)
    return {
        "time_ps": time_ps_array,
        "frame": np.arange(time_ps_array.size, dtype=float),
        "E_I_eV": np.asarray(e_i, dtype=float),
        "E_F_eV": np.asarray(e_f, dtype=float),
        "deltaE_eV": np.asarray(delta, dtype=float),
    }


def load_mlip(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], Path]:
    sources = [
        args.mlip_energy_log is not None,
        args.mlip_diagnostics is not None,
        args.mlip_xyz is not None,
    ]
    if sum(sources) != 1:
        raise ValueError("Pass exactly one of --mlip-energy-log, --mlip-diagnostics, or --mlip-xyz.")
    if args.mlip_energy_log is not None:
        data = load_mlip_energy_log(args.mlip_energy_log, mlip_ef_mode=args.mlip_ef_mode)
        return filter_max_time(data, max_time_ps=args.max_time_ps), args.mlip_energy_log
    if args.mlip_diagnostics is not None:
        data = load_mlip_diagnostics(args.mlip_diagnostics, mlip_ef_mode=args.mlip_ef_mode)
        return filter_max_time(data, max_time_ps=args.max_time_ps), args.mlip_diagnostics
    assert args.mlip_xyz is not None
    data = load_mlip_xyz(
        args.mlip_xyz,
        mlip_ef_mode=args.mlip_ef_mode,
        max_frames=args.max_mlip_xyz_frames,
    )
    return filter_max_time(data, max_time_ps=args.max_time_ps), args.mlip_xyz


def series_stats(data: dict[str, np.ndarray]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"n_samples": int(data["time_ps"].size)}
    for key in ("E_I_eV", "E_F_eV", "deltaE_eV"):
        values = data[key]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        out[f"{key}_min"] = float(np.min(values))
        out[f"{key}_max"] = float(np.max(values))
    return out


def write_summary(path: Path, aimd: dict[str, np.ndarray], mlip: dict[str, np.ndarray]) -> None:
    rows = [
        {"dataset": "AIMD", **series_stats(aimd)},
        {"dataset": "MLIP", **series_stats(mlip)},
    ]
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_energy(
    aimd: dict[str, np.ndarray],
    mlip: dict[str, np.ndarray],
    path: Path,
    *,
    aimd_label: str,
    mlip_label: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.0), sharex=False, constrained_layout=True)
    series = [
        ("E_I_eV", r"$E_I$ (eV)"),
        ("E_F_eV", r"$E_F$ (eV)"),
        ("deltaE_eV", r"$\Delta E = E_F - E_I$ (eV)"),
    ]
    for ax, (key, ylabel) in zip(axes, series, strict=True):
        ax.plot(aimd["time_ps"], aimd[key], color="#1f2937", linewidth=1.2, label=aimd_label)
        ax.plot(mlip["time_ps"], mlip[key], color="#2563eb", linewidth=1.5, label=mlip_label)
        ax.set_xlabel("time from selected window start (ps)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d4d4d8", linewidth=0.7, alpha=0.7)
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MLIP-MD endpoint energy traces against AIMD reference data."
    )
    parser.add_argument("--aimd-dat", type=Path, default=DEFAULT_AIMD_DAT)
    parser.add_argument("--aimd-last-n-frames", type=int, default=0)
    parser.add_argument(
        "--max-time-ps",
        type=float,
        default=None,
        help="Keep only samples with time_ps <= this value for both AIMD and MLIP.",
    )
    parser.add_argument("--mlip-energy-log", type=Path, default=None)
    parser.add_argument("--mlip-diagnostics", type=Path, default=None)
    parser.add_argument("--mlip-xyz", type=Path, default=None)
    parser.add_argument("--max-mlip-xyz-frames", type=int, default=None)
    parser.add_argument(
        "--mlip-ef-mode",
        choices=("with_lj", "without_lj"),
        default=None,
        help="Use MLIP ghost endpoint E_F with or without soft-core LJ correction. Default: with_lj.",
    )
    parser.add_argument(
        "--energy-mode",
        choices=("with_lj", "without_lj"),
        default=None,
        help="Deprecated alias for --mlip-ef-mode.",
    )
    parser.add_argument("--aimd-label", default="AIMD reference")
    parser.add_argument("--mlip-label", default="MLIP-MD")
    parser.add_argument("--title", default="AIMD vs MLIP-MD endpoint energies")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="T2_energy_compare")
    args = parser.parse_args()
    if not args.aimd_dat.is_file():
        raise FileNotFoundError(args.aimd_dat)
    if args.max_time_ps is not None and args.max_time_ps <= 0.0:
        raise ValueError("--max-time-ps must be positive.")
    for path in (args.mlip_energy_log, args.mlip_diagnostics, args.mlip_xyz):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    args.mlip_ef_mode = args.mlip_ef_mode or args.energy_mode or "with_lj"
    args.energy_mode = args.mlip_ef_mode
    return args


def main() -> None:
    args = parse_args()
    mlip, mlip_source = load_mlip(args)
    aimd = load_aimd_dat(
        args.aimd_dat,
        last_n_frames=args.aimd_last_n_frames if args.aimd_last_n_frames > 0 else None,
        max_time_ps=args.max_time_ps,
    )
    out_dir = args.out_dir or mlip_source.parent / "T2_energy_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{args.prefix}.png"
    summary_path = out_dir / f"{args.prefix}_summary.csv"
    plot_energy(
        aimd,
        mlip,
        plot_path,
        aimd_label=args.aimd_label,
        mlip_label=f"{args.mlip_label} (MLIP E_F {args.mlip_ef_mode})",
        title=args.title,
    )
    write_summary(summary_path, aimd, mlip)
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
