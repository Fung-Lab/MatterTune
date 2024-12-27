from matbench.bench import (  # type: ignore[reportMissingImports] # noqa
    MatbenchBenchmark,
)
from pymatgen.io.ase import AseAtomsAdaptor
import ase
import numpy as np
import mattertune.configs as MC
from mattertune.normalization import compute_per_atom_references
import json


def data_convert(structures, properties, task_name: str):
    adapter = AseAtomsAdaptor()
    atoms_list = []
    for i, structure in enumerate(structures):
        atoms = adapter.get_atoms(structure)
        assert isinstance(atoms, ase.Atoms), "Expected an Atoms object"
        atoms.info[task_name] = properties[i]
        atoms_list.append(atoms)
    return atoms_list


task_name = "matbench_mp_gap"
normalize_method = "rms"  ## "reference" or "mean_std" or "rms"

mb = MatbenchBenchmark(autoload=False, subset=[task_name])
task = list(mb.tasks)[0]
task.load()
fold = task.folds[0]

input_data, output_data = task.get_train_and_val_data(fold)

atoms_list = data_convert(input_data, output_data, task_name)

if normalize_method == "reference":
    dataset_config = MC.AtomsListDatasetConfig(atoms_list=atoms_list)
    dataset = dataset_config.create_dataset()

    ref_dict = compute_per_atom_references(
        dataset=dataset,
        property=MC.GraphPropertyConfig(
            loss=MC.MAELossConfig(),
            loss_coefficient=1.0,
            reduction="mean",
            name=task_name,
            dtype="float",
        ),
        reference_model="ridge",
    )

    filename = f"{task_name}_reference.json"
    json.dump(ref_dict, open(f"./data/{filename}", "w"), indent=4)
    print(f"Saved {task_name} reference to {filename}")
elif normalize_method == "mean_std":
    labels = [atoms.info[task_name] for atoms in atoms_list]
    mean = np.mean(labels)
    std = np.std(labels)
    filename = f"{task_name}_mean_std.json"
    json.dump({"mean": mean, "std": std}, open(f"./data/{filename}", "w"), indent=4)
    print(f"Saved {task_name} mean_std to {filename}")
elif normalize_method == "rms":
    labels = [atoms.info[task_name] for atoms in atoms_list]
    rms = np.sqrt(np.mean(np.square(labels)))
    filename = f"{task_name}_rms.json"
    json.dump({"rms": rms}, open(f"./data/{filename}", "w"), indent=4)
    print(f"Saved {task_name} rms to {filename}")
else:
    raise ValueError("Invalid normalization method")
