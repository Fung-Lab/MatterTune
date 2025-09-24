from __future__ import annotations

import json
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matbench.bench import (  # type: ignore[reportMissingImports] # noqa
    MatbenchBenchmark,
)
from pymatgen.io.ase import AseAtomsAdaptor

import mattertune.configs as MC
from mattertune.normalization import NormalizationContext
from mattertune.util import optional_import_error_message


def data_convert(task, structures, properties=None) -> list[ase.Atoms]:
    adapter = AseAtomsAdaptor()
    atoms_list = []
    for i, structure in enumerate(structures):
        atoms = adapter.get_atoms(structure)
        assert isinstance(atoms, ase.Atoms), "Expected an Atoms object"
        if properties is not None:
            atoms.info[task] = properties[i]
        atoms_list.append(atoms)
    return atoms_list


def get_normalization_ctx(
    atomic_numbers: torch.Tensor, batch_idx: torch.Tensor, num_structs: int
):
    with optional_import_error_message("torch_scatter"):
        from torch_scatter import scatter
        
    all_ones = torch.ones_like(atomic_numbers)
    num_atoms = scatter(
        all_ones,
        batch_idx,
        dim=0,
        dim_size=num_structs,
        reduce="sum",
    )

    atom_types_onehot = F.one_hot(atomic_numbers, num_classes=120)  # (n_atoms, 120)
    compositions = scatter(
        atom_types_onehot,
        batch_idx,
        dim=0,
        dim_size=num_structs,
        reduce="sum",
    )
    compositions = compositions[:, 1:]
    return NormalizationContext(num_atoms=num_atoms, compositions=compositions)


def main(args_dict: dict):
    normalize_method = args_dict["normalize_method"]
    match normalize_method:
        case "reference":
            config = MC.PerAtomReferencingNormalizerConfig(
                per_atom_references=Path(f"./data/{args_dict['task']}_reference.json")
            )
        case "mean_std":
            with open(f"./data/{args_dict['task']}_mean_std.json", "r") as f:
                mean_std = json.load(f)
                mean = mean_std["mean"]
                std = mean_std["std"]
                print(mean, std)
                print(std)
            config = MC.MeanStdNormalizerConfig(mean=mean, std=std)
        case "rms":
            with open(f"./data/{args_dict['task']}_rms.json", "r") as f:
                rms = json.load(f)["rms"]
            config = MC.RMSNormalizerConfig(rms=rms)
        case _:
            raise ValueError("Invalid normalization method")

    normalizer = config.create_normalizer_module()

    mb = MatbenchBenchmark(autoload=False, subset=[args_dict["task"]])
    task = list(mb.tasks)[0]
    task.load()
    fold = task.folds[0]
    inputs_data, outputs_data = task.get_train_and_val_data(fold)
    atoms_list = data_convert(args_dict["task"], inputs_data, outputs_data)
    atomic_numbers = torch.concat(
        [torch.tensor(atoms.get_atomic_numbers()) for atoms in atoms_list], dim=0
    ).flatten()
    batch_idx = torch.concat(
        [torch.tensor([i] * len(atoms)) for i, atoms in enumerate(atoms_list)], dim=0
    ).flatten()
    assert len(atomic_numbers) == len(batch_idx)
    values = torch.tensor(outputs_data).flatten()
    normalize_ctx = get_normalization_ctx(atomic_numbers, batch_idx, len(atoms_list))
    normalized_values = normalizer.normalize(values, normalize_ctx)

    plt.subplot(1, 2, 1)
    plt.hist(values.numpy(), bins=100, color="blue", alpha=0.5)
    plt.title("Original Distribution")
    plt.subplot(1, 2, 2)
    plt.hist(normalized_values.numpy(), bins=100, color="red", alpha=0.5)
    plt.title("Normalized Distribution")
    plt.savefig(f"./y_distribution_plots/{args_dict['task']}_{normalize_method}_distribution.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="matbench_mp_e_form")
    parser.add_argument("--normalize_method", type=str, default="mean_std")
    args = parser.parse_args()

    main(vars(args))
