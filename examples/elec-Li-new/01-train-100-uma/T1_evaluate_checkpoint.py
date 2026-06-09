from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import iread


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mattertune.main import load_finetuned_checkpoint  # noqa: E402


DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100")
TEST_DATA_ROOT = Path("/net/csefiles/coc-fung-cluster/lingyu/electrolyte")
DEFAULT_TEST_FILE = TEST_DATA_ROOT / "Li_system_test_with_del.xyz"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "local_runs" / "01-train-pair-100-uma"


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


def resolve_device(raw_device: str) -> str:
    if raw_device != "auto":
        return raw_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def structure_group(atoms: Atoms) -> str:
    return "initial" if "config_type" in atoms.info else "final"


def config_frame(atoms: Atoms) -> tuple[str, int]:
    if "config_type" in atoms.info:
        return str(atoms.info["config_type"]), int(atoms.info["frame"])
    for config_key in ("deleted_parent_config_type", "delta_pair_parent_config_type", "lambda_parent_config_type"):
        if config_key in atoms.info:
            config = str(atoms.info[config_key])
            break
    else:
        raise KeyError("Structure is missing parent config metadata.")
    for frame_key in ("deleted_parent_frame", "delta_pair_parent_frame", "lambda_source_parent_index"):
        if frame_key in atoms.info:
            frame = int(atoms.info[frame_key])
            break
    else:
        raise KeyError("Structure is missing parent frame/source-index metadata.")
    return config, frame


def predict_chunk(model: Any, atoms_chunk: list[Atoms]) -> list[dict[str, torch.Tensor]]:
    data_list = [model.atoms_to_data(atoms, has_labels=False) for atoms in atoms_chunk]
    batch = model.collate_fn(data_list)
    batch = model.batch_to_device(batch, model.device)
    predictions = model.predict_step(batch=batch, batch_idx=0)
    if len(predictions) != len(atoms_chunk):
        raise RuntimeError(
            f"Prediction count mismatch: got {len(predictions)} for {len(atoms_chunk)} structures."
        )
    return predictions


def scalar_metrics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "p95_abs": None,
            "p99_abs": None,
            "max_abs": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "mae": float(np.mean(abs_arr)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "bias": float(np.mean(arr)),
        "p95_abs": float(np.percentile(abs_arr, 95)),
        "p99_abs": float(np.percentile(abs_arr, 99)),
        "max_abs": float(np.max(abs_arr)),
    }


def force_metrics(force_errors: list[np.ndarray]) -> dict[str, float | int | None]:
    if not force_errors:
        return {
            "n_structures": 0,
            "n_components": 0,
            "component_mae_eV_A": None,
            "component_rmse_eV_A": None,
            "component_bias_eV_A": None,
        }
    arr = np.concatenate([err.reshape(-1) for err in force_errors]).astype(np.float64)
    return {
        "n_structures": len(force_errors),
        "n_components": int(arr.size),
        "component_mae_eV_A": float(np.mean(np.abs(arr))),
        "component_rmse_eV_A": float(np.sqrt(np.mean(arr**2))),
        "component_bias_eV_A": float(np.mean(arr)),
    }


def record_allowed(
    group: str,
    counts: dict[str, int],
    *,
    max_initial: int | None,
    max_final: int | None,
) -> bool:
    if group == "initial":
        return max_initial is None or counts[group] < max_initial
    return max_final is None or counts[group] < max_final


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint or find_latest_checkpoint()
    device = resolve_device(args.device)
    output_dir = args.out_dir or checkpoint.parent.parent / "T1_test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint}")
    print(f"Test file: {args.test_file}")
    print(f"Device: {device}")
    model = load_finetuned_checkpoint(str(checkpoint), map_location=torch.device(device))
    model.eval()
    model.to_device(torch.device(device))
    model.hparams.using_partition = False

    records: list[dict[str, Any]] = []
    initial_by_index: dict[int, dict[str, Any]] = {}
    counts = {"initial": 0, "final": 0}

    atoms_chunk: list[Atoms] = []
    index_chunk: list[int] = []
    for structure_index, atoms in enumerate(iread(args.test_file, ":")):
        group = structure_group(atoms)
        if not record_allowed(
            group,
            counts,
            max_initial=args.max_initial_structures,
            max_final=args.max_final_structures,
        ):
            if (
                args.max_initial_structures is not None
                and args.max_final_structures is not None
                and counts["initial"] >= args.max_initial_structures
                and counts["final"] >= args.max_final_structures
            ):
                break
            continue
        counts[group] += 1
        atoms_chunk.append(atoms)
        index_chunk.append(structure_index)
        if len(atoms_chunk) == args.batch_size:
            collect_predictions(model, atoms_chunk, index_chunk, records, initial_by_index)
            atoms_chunk = []
            index_chunk = []

    if atoms_chunk:
        collect_predictions(model, atoms_chunk, index_chunk, records, initial_by_index)

    delta_rows: list[dict[str, Any]] = []
    for record in records:
        if record["endpoint"] != "final":
            continue
        parent_index = record.get("deleted_parent_index")
        if parent_index is None or int(parent_index) not in initial_by_index:
            continue
        initial = initial_by_index[int(parent_index)]
        dft_delta = record["energy_gt_eV"] - initial["energy_gt_eV"]
        pred_delta = record["energy_pred_eV"] - initial["energy_pred_eV"]
        delta_rows.append(
            {
                "config_type": record["config_type"],
                "frame": record["frame"],
                "initial_structure_index": int(parent_index),
                "final_structure_index": record["structure_index"],
                "dft_delta_E_eV": dft_delta,
                "pred_delta_E_eV": pred_delta,
                "delta_E_error_eV": pred_delta - dft_delta,
                "initial_energy_error_eV": initial["energy_error_eV"],
                "final_energy_error_eV": record["energy_error_eV"],
            }
        )

    initial_records = [row for row in records if row["endpoint"] == "initial"]
    final_records = [row for row in records if row["endpoint"] == "final"]
    summary = {
        "checkpoint": str(checkpoint),
        "test_file": str(args.test_file),
        "n_records": len(records),
        "E_I_error_eV": scalar_metrics([row["energy_error_eV"] for row in initial_records]),
        "E_F_error_eV": scalar_metrics([row["energy_error_eV"] for row in final_records]),
        "F_I_error": force_metrics([row["_force_error"] for row in initial_records]),
        "F_F_error": force_metrics([row["_force_error"] for row in final_records]),
        "delta_E_error_eV": scalar_metrics([row["delta_E_error_eV"] for row in delta_rows]),
        "by_config_type": summarize_by_config(records, delta_rows),
    }

    structure_csv = output_dir / f"{args.prefix}_structures.csv"
    delta_csv = output_dir / f"{args.prefix}_delta_pairs.csv"
    summary_json = output_dir / f"{args.prefix}_summary.json"
    write_structure_csv(structure_csv, records)
    write_delta_csv(delta_csv, delta_rows)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {structure_csv}")
    print(f"Wrote {delta_csv}")
    print(f"Wrote {summary_json}")
    print(json.dumps(summary["delta_E_error_eV"], indent=2))
    return summary


def collect_predictions(
    model: Any,
    atoms_chunk: list[Atoms],
    structure_indices: list[int],
    records: list[dict[str, Any]],
    initial_by_index: dict[int, dict[str, Any]],
) -> None:
    predictions = predict_chunk(model, atoms_chunk)
    for structure_index, atoms, pred in zip(structure_indices, atoms_chunk, predictions, strict=True):
        endpoint = structure_group(atoms)
        config_type, frame = config_frame(atoms)
        gt_e = float(atoms.get_potential_energy())
        gt_f = np.asarray(atoms.get_forces(), dtype=np.float64)
        ## if not 0-dimension, convert to 0-dimension array and extract the value, to avoid potential issues with 0-dim tensors in later calculations.
        if isinstance(pred["energy"], torch.Tensor) and pred["energy"].ndim != 0:
            pred_e = float(pred["energy"].detach().cpu().numpy().item())
        else:
            pred_e = float(pred["energy"].detach().cpu().numpy())
        pred_f = np.asarray(pred["forces"].detach().cpu().numpy(), dtype=np.float64)
        force_error = pred_f - gt_f

        row: dict[str, Any] = {
            "structure_index": int(structure_index),
            "endpoint": endpoint,
            "config_type": config_type,
            "frame": int(frame),
            "natoms": len(atoms),
            "energy_gt_eV": gt_e,
            "energy_pred_eV": pred_e,
            "energy_error_eV": pred_e - gt_e,
            "force_component_mae_eV_A": float(np.mean(np.abs(force_error))),
            "force_component_rmse_eV_A": float(np.sqrt(np.mean(force_error**2))),
            "force_component_bias_eV_A": float(np.mean(force_error)),
            "_force_error": force_error,
        }
        if endpoint == "final":
            row["deleted_parent_index"] = int(atoms.info["deleted_parent_index"])
            row["deleted_train_delete_index"] = int(atoms.info["deleted_train_delete_index"])
        else:
            initial_by_index[int(structure_index)] = row
        records.append(row)


def summarize_by_config(
    records: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_delta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped_records[str(row["config_type"])].append(row)
    for row in delta_rows:
        grouped_delta[str(row["config_type"])].append(row)

    out: dict[str, Any] = {}
    for config_type in sorted(set(grouped_records) | set(grouped_delta)):
        rows = grouped_records.get(config_type, [])
        initial = [row for row in rows if row["endpoint"] == "initial"]
        final = [row for row in rows if row["endpoint"] == "final"]
        deltas = grouped_delta.get(config_type, [])
        out[config_type] = {
            "E_I_error_eV": scalar_metrics([row["energy_error_eV"] for row in initial]),
            "E_F_error_eV": scalar_metrics([row["energy_error_eV"] for row in final]),
            "F_I_error": force_metrics([row["_force_error"] for row in initial]),
            "F_F_error": force_metrics([row["_force_error"] for row in final]),
            "delta_E_error_eV": scalar_metrics([row["delta_E_error_eV"] for row in deltas]),
        }
    return out


def write_structure_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "structure_index",
        "endpoint",
        "config_type",
        "frame",
        "natoms",
        "energy_gt_eV",
        "energy_pred_eV",
        "energy_error_eV",
        "force_component_mae_eV_A",
        "force_component_rmse_eV_A",
        "force_component_bias_eV_A",
        "deleted_parent_index",
        "deleted_train_delete_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            writer.writerow(clean)


def write_delta_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "config_type",
        "frame",
        "initial_structure_index",
        "final_structure_index",
        "dft_delta_E_eV",
        "pred_delta_E_eV",
        "delta_E_error_eV",
        "initial_energy_error_eV",
        "final_energy_error_eV",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on E_I, E_F, F_I, F_F, and delta E."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="T1_test_eval")
    parser.add_argument("--max-initial-structures", type=int, default=None)
    parser.add_argument("--max-final-structures", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    for path in (args.test_file,):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    return args


if __name__ == "__main__":
    evaluate(parse_args())
