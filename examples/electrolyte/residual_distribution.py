"""Residual energy vs per-atom reference: CSV summary + box plots by config_type.

Energy reference JSON: same format as MatterTune training (`{ "Z": ref_eV, ... }`).
Residual (total): E_atoms - sum_z n_z * ref_z  (units: eV, same as extxyz labels).

Figures: one PNG per (carrier cation, anion, solvent) triple present in the data.
Default grid uses 5 cations × 1 anion (FSI) × 8 solvents; empty triples are skipped.
"""
from __future__ import annotations
from ase.io import iread
import numpy as np
import matplotlib.pyplot as plt

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Defaults — edit here, set env vars, or use CLI flags.
# ---------------------------------------------------------------------------
_EXAMPLE_DIR = Path(__file__).resolve().parent

_DATA_ROOT = Path(
    os.environ.get(
        "DATA_ROOT",
        "/net/csefiles/coc-fung-cluster/lingyu/electrolyte",
    )
)

# Per-atom reference JSON (eV per atom for each atomic number Z)
PER_ATOM_ENERGY_REFERENCE_JSON: Path = Path(
    os.environ.get(
        "ENERGY_REFERENCE",
        str(_DATA_ROOT / "train-mace-residual-energy_reference.json"),
    )
)

# Full test / validation trajectory bundle in extended XYZ
TEST_XYZ_PATH: Path = Path(
    os.environ.get(
        "TEST_FILE",
        str(_DATA_ROOT / "test.xyz"),
    )
)

# Where to write residual_distribution.csv and PNGs
RESIDUAL_DIST_OUTPUT_DIR: Path = Path(
    os.environ.get(
        "RESIDUAL_DIST_OUTPUT_DIR",
        str(_EXAMPLE_DIR / "results" / "residual_distribution"),
    )
)

# Plot layout: iterate these orders so filenames are stable (5 × |anions| × 8)
CATION_ORDER: tuple[str, ...] = ("Li", "Na", "K", "Mg", "Zn")
ANION_ORDER: tuple[str, ...] = ("FSI",)
SOLVENT_ORDER: tuple[str, ...] = (
    "G2",
    "DME",
    "SFL",
    "PC",
    "EC",
    "FEC",
    "FEMC",
    "THF",
)

_CONFIG_TYPE_RE = re.compile(
    r"^case\d+-case\d+-(?P<cation>[^-]+)-(?P<anion>[^-]+)-(?P<solvent>[^-]+)-"
)


def load_reference_eV(path: Path) -> dict[int, float]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[int, float] = {}
    for k, v in raw.items():
        out[int(k)] = float(v)
    return out


def reference_energy_total(atoms, refs: dict[int, float]) -> float:
    numbers = atoms.get_atomic_numbers()
    acc = 0.0
    for z in np.unique(numbers):
        n = int(np.sum(numbers == z))
        if z not in refs:
            raise KeyError(
                f"Atomic number Z={z} missing from reference JSON "
                f"(have keys up to {max(refs)})"
            )
        acc += n * refs[z]
    return acc


def parse_chem_triple(config_type: str | None) -> tuple[str, str, str] | None:
    if not config_type:
        return None
    m = _CONFIG_TYPE_RE.match(config_type.strip())
    if not m:
        return None
    return m.group("cation"), m.group("anion"), m.group("solvent")


def collect_residuals(
    xyz_path: Path,
    refs: dict[int, float],
) -> dict[str, list[float]]:
    """config_type -> list of residual energies (eV, whole structure)."""
    by_ct: dict[str, list[float]] = defaultdict(list)
    n_missing = 0
    n_total = 0
    for atoms in iread(str(xyz_path)):
        n_total += 1
        ct = atoms.info.get("config_type")
        if not isinstance(ct, str):
            n_missing += 1
            continue
        e = atoms.get_potential_energy()
        e_ref = reference_energy_total(atoms, refs)
        by_ct[ct].append(float(e - e_ref))
    if n_missing:
        print(
            f"[warn] {n_missing}/{n_total} frames missing config_type; skipped.")
    return by_ct


def write_summary_csv(
    by_ct: dict[str, list[float]],
    out_csv: Path,
) -> None:
    rows = []
    for ct, values in sorted(by_ct.items()):
        arr = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "config_type": ct,
                "n": len(arr),
                "min_eV": float(arr.min()),
                "max_eV": float(arr.max()),
                "mean_eV": float(arr.mean()),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["config_type", "n", "min_eV", "max_eV", "mean_eV"],
        )
        w.writeheader()
        w.writerows(rows)


def plot_group_boxplots(
    *,
    cation: str,
    anion: str,
    solvent: str,
    by_ct: dict[str, list[float]],
    out_png: Path,
    y_label: str = "Residual energy (eV)",
) -> bool:
    """Boxplot for all config_types matching (cation, anion, solvent). Returns False if empty."""
    matching = {
        ct: vals
        for ct, vals in by_ct.items()
        if parse_chem_triple(ct) == (cation, anion, solvent)
    }
    if not matching:
        return False
    labels = sorted(matching.keys())
    data = [matching[lbl] for lbl in labels]

    n = len(labels)
    fig_w = min(56.0, max(10.0, 0.55 * n + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=True,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor="lightsteelblue",
                  edgecolor="steelblue", alpha=0.85)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel(y_label)
    ax.set_xlabel("config_type")
    ax.set_title(f"{cation} / {anion} / {solvent}")
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right", fontsize=7)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def run(
    *,
    ref_path: Path,
    xyz_path: Path,
    out_dir: Path,
    cation_order: tuple[str, ...] = CATION_ORDER,
    anion_order: tuple[str, ...] = ANION_ORDER,
    solvent_order: tuple[str, ...] = SOLVENT_ORDER,
) -> None:
    refs = load_reference_eV(ref_path)
    print(
        f"Loaded reference for Z in [{min(refs)}, {max(refs)}] from {ref_path}")
    by_ct = collect_residuals(xyz_path, refs)
    print(
        f"Unique config_type: {len(by_ct)}  |  structures: {sum(map(len, by_ct.values()))}")

    out_dir = Path(out_dir)
    write_summary_csv(by_ct, out_dir / "residual_summary.csv")
    print(f"Wrote {out_dir / 'residual_summary.csv'}")

    n_plots = 0
    for c in cation_order:
        for a in anion_order:
            for s in solvent_order:
                safe = f"{c}_{a}_{s}".replace(" ", "_")
                png = out_dir / f"residual_box_{safe}.png"
                if plot_group_boxplots(
                    cation=c,
                    anion=a,
                    solvent=s,
                    by_ct=by_ct,
                    out_png=png,
                ):
                    n_plots += 1
                    print(f"Wrote {png}")
    print(f"Done. {n_plots} non-empty boxplot figures under {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference",
        type=Path,
        default=PER_ATOM_ENERGY_REFERENCE_JSON,
        help="Per-atom reference JSON (eV)",
    )
    p.add_argument(
        "--xyz",
        type=Path,
        default=TEST_XYZ_PATH,
        help="Extended XYZ with energy + config_type in frame comments",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=RESIDUAL_DIST_OUTPUT_DIR,
        help="Output directory for CSV and PNGs",
    )
    args = p.parse_args()
    run(ref_path=args.reference, xyz_path=args.xyz, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
