from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.io import iread
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mattertune.main import load_finetuned_checkpoint  # noqa: E402


PRED_BATCH_SIZE = 4
FORCE_PLOT_MAX_POINTS = 500_000
RNG_SEED = 20260706


@dataclass
class RunningMetrics:
    energy_abs_mev_atom: float = 0.0
    energy_sq_mev_atom2: float = 0.0
    n_structures: int = 0
    force_abs_mev_a: float = 0.0
    force_sq_mev_a2: float = 0.0
    n_force_components: int = 0

    def update_energy(self, error_mev_atom: np.ndarray) -> None:
        self.energy_abs_mev_atom += float(np.abs(error_mev_atom).sum())
        self.energy_sq_mev_atom2 += float(np.square(error_mev_atom).sum())
        self.n_structures += int(error_mev_atom.size)

    def update_forces(self, error_mev_a: np.ndarray) -> None:
        flat = error_mev_a.reshape(-1)
        self.force_abs_mev_a += float(np.abs(flat).sum())
        self.force_sq_mev_a2 += float(np.square(flat).sum())
        self.n_force_components += int(flat.size)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_structures": self.n_structures,
            "n_force_components": self.n_force_components,
            "energy_mae_meV_per_atom": self.energy_abs_mev_atom / self.n_structures,
            "energy_rmse_meV_per_atom": float(
                np.sqrt(self.energy_sq_mev_atom2 / self.n_structures)
            ),
            "forces_mae_meV_per_A": self.force_abs_mev_a / self.n_force_components,
            "forces_rmse_meV_per_A": float(
                np.sqrt(self.force_sq_mev_a2 / self.n_force_components)
            ),
        }


class ForceReservoir:
    def __init__(self, max_points: int, seed: int) -> None:
        self.max_points = max_points
        self.rng = np.random.default_rng(seed)
        self.keys = np.empty((0,), dtype=np.float64)
        self.gt = np.empty((0,), dtype=np.float32)
        self.pred = np.empty((0,), dtype=np.float32)
        self.n_seen = 0

    def update(self, force_gt: np.ndarray, force_pred: np.ndarray) -> None:
        gt = np.asarray(force_gt, dtype=np.float32).reshape(-1)
        pred = np.asarray(force_pred, dtype=np.float32).reshape(-1)
        if gt.size == 0:
            return
        self.n_seen += int(gt.size)
        new_keys = self.rng.random(gt.size)

        keys = np.concatenate([self.keys, new_keys])
        gt_all = np.concatenate([self.gt, gt])
        pred_all = np.concatenate([self.pred, pred])

        if keys.size > self.max_points:
            keep = np.argpartition(keys, self.max_points - 1)[: self.max_points]
            keys = keys[keep]
            gt_all = gt_all[keep]
            pred_all = pred_all[keep]

        self.keys = keys
        self.gt = gt_all
        self.pred = pred_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a MatterTune checkpoint on an extxyz file and save energy/force "
            "parity plots plus MAE/RMSE metrics."
        )
    )
    parser.add_argument("xyz_path", type=Path)
    parser.add_argument("ckpt_path", type=Path)
    return parser.parse_args()


def default_output_dir(xyz_path: Path, ckpt_path: Path) -> Path:
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    return run_dir / "parity_eval" / xyz_path.stem


def atoms_energy(atoms: Atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        for key in ("energy", "Energy", "dft_energy", "DFT_energy"):
            if key in atoms.info:
                return float(atoms.info[key])
        raise


def atoms_forces(atoms: Atoms) -> np.ndarray:
    try:
        return np.asarray(atoms.get_forces(), dtype=np.float64)
    except Exception:
        if "forces" in atoms.arrays:
            return np.asarray(atoms.arrays["forces"], dtype=np.float64)
        raise


def predict_batch(model, atoms_batch: list[Atoms], device: torch.device) -> list[dict[str, torch.Tensor]]:
    data_list = [model.atoms_to_data(atoms, has_labels=False) for atoms in atoms_batch]
    batch = model.collate_fn(data_list)
    batch = model.batch_to_device(batch, device)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        predictions = model.predict_step(batch, batch_idx=0)
    return predictions


def append_per_structure_rows(
    csv_writer: csv.writer,
    start_index: int,
    natoms: np.ndarray,
    energy_gt: np.ndarray,
    energy_pred: np.ndarray,
    energy_error_mev_atom: np.ndarray,
) -> None:
    for offset, (natom, gt, pred, err) in enumerate(
        zip(natoms, energy_gt, energy_pred, energy_error_mev_atom, strict=True)
    ):
        csv_writer.writerow(
            [
                start_index + offset,
                int(natom),
                f"{gt:.12g}",
                f"{pred:.12g}",
                f"{err:.12g}",
            ]
        )


def update_parity_limits(ax, x: np.ndarray, y: np.ndarray) -> None:
    values = np.concatenate([np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    lo = float(values.min())
    hi = float(values.max())
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1.0)
    else:
        pad = 0.03 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")


def save_parity_plot(
    output_path: Path,
    energy_gt: np.ndarray,
    energy_pred: np.ndarray,
    force_sample: ForceReservoir,
    metrics: dict[str, float | int],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.scatter(energy_gt, energy_pred, s=7, alpha=0.45, rasterized=True)
    update_parity_limits(ax, energy_gt, energy_pred)
    ax.set_xlabel("Ground truth energy (eV)")
    ax.set_ylabel("Predicted energy (eV)")
    ax.set_title(
        "Energy parity\n"
        f"MAE {metrics['energy_mae_meV_per_atom']:.3f} meV/atom, "
        f"RMSE {metrics['energy_rmse_meV_per_atom']:.3f} meV/atom"
    )

    ax = axes[1]
    ax.scatter(force_sample.gt, force_sample.pred, s=1, alpha=0.18, rasterized=True)
    update_parity_limits(ax, force_sample.gt, force_sample.pred)
    ax.set_xlabel("Ground truth force component (eV/A)")
    ax.set_ylabel("Predicted force component (eV/A)")
    ax.set_title(
        "Force parity\n"
        f"MAE {metrics['forces_mae_meV_per_A']:.3f} meV/A, "
        f"RMSE {metrics['forces_rmse_meV_per_A']:.3f} meV/A"
    )
    ax.text(
        0.02,
        0.98,
        f"plotted {len(force_sample.gt):,} / {force_sample.n_seen:,} components",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    xyz_path = args.xyz_path.resolve()
    ckpt_path = args.ckpt_path.resolve()
    if not xyz_path.is_file():
        raise FileNotFoundError(xyz_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    output_dir = default_output_dir(xyz_path, ckpt_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_finetuned_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    model.to_device(device)
    model.hparams.using_partition = False
    print(f"Using device: {device}")
    print(f"Reading xyz stream: {xyz_path}")
    print(f"Output directory: {output_dir}")

    metrics = RunningMetrics()
    force_sample = ForceReservoir(FORCE_PLOT_MAX_POINTS, RNG_SEED)
    energy_gt_all: list[float] = []
    energy_pred_all: list[float] = []

    per_structure_path = output_dir / "per_structure_errors.csv"
    with per_structure_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "structure_index",
                "natoms",
                "energy_gt_eV",
                "energy_pred_eV",
                "energy_error_meV_per_atom",
            ]
        )

        atoms_batch: list[Atoms] = []
        start_index = 0
        pbar = tqdm(desc="Evaluating structures", unit="structure")
        for atoms in iread(xyz_path, index=":"):
            atoms_batch.append(atoms)
            if len(atoms_batch) < PRED_BATCH_SIZE:
                continue
            start_index = evaluate_atoms_batch(
                model,
                device,
                atoms_batch,
                start_index,
                writer,
                metrics,
                force_sample,
                energy_gt_all,
                energy_pred_all,
            )
            pbar.update(len(atoms_batch))
            atoms_batch = []
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if atoms_batch:
            start_index = evaluate_atoms_batch(
                model,
                device,
                atoms_batch,
                start_index,
                writer,
                metrics,
                force_sample,
                energy_gt_all,
                energy_pred_all,
            )
            pbar.update(len(atoms_batch))
        pbar.close()

    metric_dict = metrics.as_dict()
    metric_dict.update(
        {
            "xyz_path": str(xyz_path),
            "ckpt_path": str(ckpt_path),
            "device": str(device),
            "prediction_batch_size": PRED_BATCH_SIZE,
            "force_plot_points": int(len(force_sample.gt)),
            "force_components_total": int(force_sample.n_seen),
        }
    )

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metric_dict, handle, indent=2, sort_keys=True)

    plot_path = output_dir / "parity_scatter.png"
    save_parity_plot(
        plot_path,
        np.asarray(energy_gt_all, dtype=np.float64),
        np.asarray(energy_pred_all, dtype=np.float64),
        force_sample,
        metric_dict,
    )

    print(json.dumps(metric_dict, indent=2, sort_keys=True))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved parity plot: {plot_path}")
    print(f"Saved per-structure table: {per_structure_path}")


def evaluate_atoms_batch(
    model,
    device: torch.device,
    atoms_batch: list[Atoms],
    start_index: int,
    writer: csv.writer,
    metrics: RunningMetrics,
    force_sample: ForceReservoir,
    energy_gt_all: list[float],
    energy_pred_all: list[float],
) -> int:
    predictions = predict_batch(model, atoms_batch, device)

    natoms = np.asarray([len(atoms) for atoms in atoms_batch], dtype=np.float64)
    energy_gt = np.asarray([atoms_energy(atoms) for atoms in atoms_batch], dtype=np.float64)
    force_gt = [atoms_forces(atoms) for atoms in atoms_batch]
    energy_pred = np.asarray(
        [float(pred["energy"].detach().cpu().item()) for pred in predictions],
        dtype=np.float64,
    )
    force_pred = [
        np.asarray(pred["forces"].detach().cpu(), dtype=np.float64) for pred in predictions
    ]

    energy_error_mev_atom = (energy_pred - energy_gt) / natoms * 1000.0
    metrics.update_energy(energy_error_mev_atom)
    append_per_structure_rows(
        writer,
        start_index,
        natoms,
        energy_gt,
        energy_pred,
        energy_error_mev_atom,
    )
    energy_gt_all.extend(energy_gt.tolist())
    energy_pred_all.extend(energy_pred.tolist())

    for gt, pred in zip(force_gt, force_pred, strict=True):
        if gt.shape != pred.shape:
            raise ValueError(f"Force shape mismatch: ground truth {gt.shape}, predicted {pred.shape}")
        metrics.update_forces((pred - gt) * 1000.0)
        force_sample.update(gt, pred)

    return start_index + len(atoms_batch)


if __name__ == "__main__":
    main()
