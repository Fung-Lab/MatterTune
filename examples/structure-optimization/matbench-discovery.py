from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import ase
import nshconfig as C
import numpy as np
import pandas as pd
from ase.calculators.calculator import Calculator
from ase.filters import ExpCellFilter, FrechetCellFilter, UnitCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key
from tqdm.auto import tqdm
from typing_extensions import NotRequired, TypedDict

from mattertune.backbones import JMPBackboneModule

log = logging.getLogger(__name__)

FILTER_CLS = {
    "frechet": FrechetCellFilter,
    "unit": UnitCellFilter,
    "exp": ExpCellFilter,
}
OPTIM_CLS = {"FIRE": FIRE, "LBFGS": LBFGS, "BFGS": BFGS}


class RelaxerConfig(C.Config):
    optimizer: Literal["FIRE", "LBFGS", "BFGS"]
    """ASE optimizer to use for relaxation."""

    optimizer_kwargs: dict[str, Any] = {}
    """Keyword arguments to pass to the optimizer."""

    force_max: float
    """Maximum force allowed during relaxation."""

    max_steps: int
    """Maximum number of relaxation steps."""

    cell_filter: Literal["frechet", "exp", "unit"] | None = None
    """Cell filter to use for relaxation."""

    optim_log_file: Path = Path("/dev/null")
    """Path to the log file for the optimizer. If None, the log file will be written to /dev/null."""

    output_relaxed_structures: bool = True
    """Whether to output the relaxed structures. This can be memory/storage intensive."""

    def _cell_filter_cls(self):
        if self.cell_filter is None:
            return None
        return FILTER_CLS[self.cell_filter]

    def _optim_cls(self):
        return OPTIM_CLS[self.optimizer]


class DatasetItem(TypedDict):
    material_id: str
    """Material ID of the structure."""

    atoms: ase.Atoms
    """ase.Atoms object representing the structure."""

    metadata: dict[str, Any]
    """Metadata associated with the structure, will be saved with the relaxation results."""


class RelaxResult(TypedDict):
    material_id: str
    """Material ID of the structure."""

    energy: float
    """Relaxed energy (raw model output)."""

    e_form_per_atom: float
    """Formation energy per atom (corrected using mp_elemental_ref_energies)."""

    hull_true: NotRequired[float]
    """Ground-truth energy above hull."""

    hull_pred: NotRequired[float]
    """Predicted energy above hull."""

    metadata: dict[str, Any]
    """Metadata associated with the structure, will be saved with the relaxation results."""

    structure: NotRequired[dict[str, Any]]
    """Relaxed structure."""


def _load_dataset(
    subset: float | int | None,
    subset_seed: int,
    rank: int = 0,
    world_size: int = 1,
):
    from matbench_discovery.data import DataFiles

    # Load the DataFrames
    df_wbm = pd.read_csv(DataFiles.wbm_summary.path)
    df_wbm_initial = pd.read_json(DataFiles.wbm_initial_structures.path)

    # Split the dataset rows
    rows = np.arange(len(df_wbm_initial))
    rows_split = np.array_split(rows, world_size)

    # Get the rows for this rank
    row_idxs = rows_split[rank]
    df_wbm = df_wbm.iloc[row_idxs]
    df_wbm_initial = df_wbm_initial.iloc[row_idxs]

    # Subset the dataset
    if subset is not None:
        if isinstance(subset, float):
            subset = int(subset * len(df_wbm))

        np.random.seed(subset_seed)
        subset_idxs = np.random.choice(len(df_wbm), subset, replace=False)
        df_wbm = df_wbm.iloc[subset_idxs]
        df_wbm_initial = df_wbm_initial.iloc[subset_idxs]
        log.info(f"Subset the dataset to {len(df_wbm)} rows.")
    else:
        log.info(f"Using the entire dataset with {len(df_wbm)} rows.")

    # Merge on matching material ids
    df = pd.merge(df_wbm, df_wbm_initial, on=Key.mat_id.value, how="inner")

    return df


def _dataset_generator(df: pd.DataFrame):
    for idx, row in df.iterrows():
        # Get the material id
        material_id = row[Key.mat_id.value]
        # Get the initial structure
        atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(row["initial_structure"]))
        assert isinstance(atoms, ase.Atoms), f"Expected ase.Atoms, got {type(atoms)}"
        # Get everything else, except the initial structure, as a dictionary.
        metadata = row.drop("initial_structure").to_dict()
        # Add the row index to the metadata
        metadata["__row_idx__"] = idx

        # Create the dataset item
        dataset_item: DatasetItem = {
            "material_id": material_id,
            "atoms": atoms,
            "metadata": metadata,
        }

        yield dataset_item


def _get_composition(atoms: ase.Atoms) -> dict[int, int]:
    """Get atomic composition as a mapping of atomic numbers to counts.

    Args:
        atoms: ASE Atoms object

    Returns:
        Dictionary mapping atomic numbers to their counts
    """
    return dict(Counter(atoms.get_atomic_numbers()))


def _compute_e_form_per_atom(energy: float, atoms: ase.Atoms) -> float:
    """Calculate formation energy per atom.

    Args:
        energy: Total energy of the structure
        atoms: ASE Atoms object

    Returns:
        Formation energy per atom
    """
    from matbench_discovery.energy import get_e_form_per_atom

    composition = _get_composition(atoms)
    e_form_per_atom = get_e_form_per_atom(
        {"energy": energy, "composition": composition}
    )
    return e_form_per_atom


def _relax_single(
    dataset_item: DatasetItem,
    *,
    config: RelaxerConfig,
    calculator: Calculator,
    relaxed: set[str],
) -> RelaxResult | None:
    """Run WBM relaxation on a single structure using an ASE optimizer."""

    # Resolve the optimizer and cell filter classes
    optim_cls = config._optim_cls()
    filter_cls = config._cell_filter_cls()

    material_id = dataset_item["material_id"]
    if material_id in relaxed:
        logging.info(f"Structure {material_id} has already been relaxed.")
        return None

    atoms = dataset_item["atoms"]
    try:
        atoms.calc = calculator

        if filter_cls is not None:
            optim = optim_cls(
                filter_cls(atoms),
                logfile=str(config.optim_log_file),
                **config.optimizer_kwargs,
            )
        else:
            optim = optim_cls(
                atoms,
                logfile=str(config.optim_log_file),
                **config.optimizer_kwargs,
            )

        optim.run(fmax=config.force_max, steps=config.max_steps)

        energy = atoms.get_potential_energy()

        # Calculate formation energy per atom
        e_form_per_atom = _compute_e_form_per_atom(energy, atoms)

        # If the metadata has hull, we can compute this as well.
        from matbench_discovery.enums import MbdKey

        hull_results: Any = {}
        if (hull_true := dataset_item["metadata"].get(MbdKey.each_true)) is not None:
            assert (
                e_form_true := dataset_item["metadata"].get(MbdKey.e_form_wbm)
            ) is not None, "Expected e_form_true to be present if hull_true is present."

            hull_pred = hull_true + (e_form_per_atom - e_form_true)
            hull_results = {"hull_true": hull_true, "hull_pred": hull_pred}

        # Create the result
        result: RelaxResult = {
            "material_id": material_id,
            "energy": energy,
            "e_form_per_atom": e_form_per_atom,
            "metadata": dataset_item["metadata"],
            **hull_results,
        }
        if config.output_relaxed_structures:
            result["structure"] = AseAtomsAdaptor.get_structure(atoms).as_dict()

        relaxed.add(material_id)
        return result
    except Exception:
        logging.exception(f"Failed to relax {material_id}")
        return None


def _run_relaxation(
    dataset: Iterable[DatasetItem],
    *,
    config: RelaxerConfig,
    calculator: Calculator,
) -> pd.DataFrame:
    """Run relaxation on multiple structures and collect results in a DataFrame."""
    relaxed: set[str] = set()
    results = []

    for item in tqdm(dataset, desc="Relaxing structures"):
        if (
            result := _relax_single(
                item, config=config, calculator=calculator, relaxed=relaxed
            )
        ) is None:
            continue

        results.append(result)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Set the index to the material id
    df.set_index("material_id", inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint file of the MatterTuner model.",
    )
    parser.add_argument(
        "--subset",
        type=float,
        help="Fraction of the dataset to use. If None, the entire dataset will be used.",
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=0,
        help="Seed for the subset selection.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of the process in the distributed setup.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes in the distributed setup.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=list(OPTIM_CLS.keys()),
        help="ASE optimizer to use for relaxation.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        required=True,
        help="Maximum force allowed during relaxation.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        required=True,
        help="Maximum number of relaxation steps.",
    )
    parser.add_argument(
        "--cell_filter",
        type=str,
        choices=list(FILTER_CLS.keys()),
        help="Cell filter to use for relaxation.",
    )
    parser.add_argument(
        "--output_relaxed_structures",
        action="store_true",
        help="Whether to output the relaxed structures.",
    )
    parser.add_argument(
        "--optim_log_file",
        type=Path,
        default=Path("/dev/null"),
        help="Path to the log file for the optimizer.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("relaxation_results.csv"),
        help="Path to the output file.",
    )
    args = parser.parse_args()

    # Create the relaxation configuration
    config = RelaxerConfig(
        optimizer=args.optimizer,
        force_max=args.fmax,
        max_steps=args.max_steps,
        cell_filter=args.cell_filter,
        output_relaxed_structures=args.output_relaxed_structures,
        optim_log_file=args.optim_log_file,
    )

    # Load the dataset
    df = _load_dataset(
        args.subset,
        args.subset_seed,
        args.rank,
        args.world_size,
    )

    # Create the calculator
    module = JMPBackboneModule.load_from_checkpoint(args.ckpt)
    calculator = module.ase_calculator()

    # Run relaxation
    dataset = _dataset_generator(df)
    df_results = _run_relaxation(dataset, config=config, calculator=calculator)

    # Save the results
    df_results.to_csv(args.output_file)
    log.info(f"Relaxation results saved to {args.output_file}")


if __name__ == "__main__":
    main()
