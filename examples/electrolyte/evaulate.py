from __future__ import annotations

import rich
from ase import Atoms
from ase.io import read
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from mattertune.main import load_pretrained_model
from mattertune.main import load_finetuned_checkpoint

# model_type = "mattersim"
# pt_model_name = "MatterSim-v1.0.0-1M"
# pt_model = load_pretrained_model(
#     model_type=model_type,
#     model_name=pt_model_name,
#     device=f"cuda:0"
# )

ft_model_name = "/net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/20260425-224420-mattersim/checkpoints/MatterSim-v1.0.0-1M-best.ckpt"
ft_model = load_finetuned_checkpoint(ft_model_name)


# val_atoms_list: list[Atoms] = read(
#     "/net/csefiles/coc-fung-cluster/lingyu/electrolyte/Li-system-train-ase.xyz", ":")  # type: ignore

# calc = pt_model.ase_calculator()
# gt_energies_per_atom = []
# gt_forces = []
# pred_energies_per_atom = []
# pred_forces = []
# for atoms in tqdm(val_atoms_list):
#     n = len(atoms)
#     gt_e = atoms.get_potential_energy()
#     gt_f = atoms.get_forces()
#     atoms.set_calculator(calc)
#     pred_e = atoms.get_potential_energy()
#     pred_f = atoms.get_forces()
#     gt_energies_per_atom.append(gt_e / n)
#     gt_forces.extend(np.array(gt_f).tolist())
#     pred_energies_per_atom.append(pred_e / n)
#     pred_forces.extend(np.array(pred_f).tolist())

# e_mae = torch.nn.L1Loss()(torch.tensor(gt_energies_per_atom),
#                           torch.tensor(pred_energies_per_atom))
# f_mae = torch.nn.L1Loss()(torch.tensor(gt_forces), torch.tensor(pred_forces))
# e_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(
#     gt_energies_per_atom), torch.tensor(pred_energies_per_atom)))
# f_rmse = torch.sqrt(torch.nn.MSELoss()(
#     torch.tensor(gt_forces), torch.tensor(pred_forces)))
# rich.print(f"Energy MAE: {e_mae} eV/atom")
# rich.print(f"Forces MAE: {f_mae} eV/Ang")
# rich.print(f"Energy RMSE: {e_rmse} eV/atom")
# rich.print(f"Forces RMSE: {f_rmse} eV/Ang")

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(gt_energies_per_atom, pred_energies_per_atom)
# plt.xlabel("True Energy (eV/atom)")
# plt.ylabel("Predicted Energy (eV/atom)")
# plt.title("Energy MAE: {e_mae} eV/atom")
# plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes,
#          linestyle="-", color="k", alpha=0.7)
# plt.subplot(1, 2, 2)
# plt.scatter(gt_forces, pred_forces)
# plt.xlabel("True Forces (eV/Ang)")
# plt.ylabel("Predicted Forces (eV/Ang)")
# plt.title("Forces MAE: {f_mae} eV/Ang")
# plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes,
#          linestyle="-", color="k", alpha=0.7)
# plt.tight_layout()
# plt.savefig("./results/pt_parity_plot.png")
# plt.close()

calc = ft_model.ase_calculator(
    device=f"cuda:1"
)
val_atoms_list: list[Atoms] = read(
    "/net/csefiles/coc-fung-cluster/lingyu/electrolyte/test.xyz", ":")  # type: ignore
random_indices = np.random.choice(len(val_atoms_list), 1000, replace=False)
val_atoms_list = [val_atoms_list[i] for i in random_indices]
gt_energies = []
gt_energies_per_atom = []
gt_forces = []
pred_energies = []
pred_energies_per_atom = []
pred_forces = []
for atoms in tqdm(val_atoms_list):
    n = len(atoms)
    gt_e = atoms.get_potential_energy()
    gt_f = atoms.get_forces()
    atoms.set_calculator(calc)
    pred_e = atoms.get_potential_energy()
    pred_f = atoms.get_forces()
    gt_energies_per_atom.append(gt_e / n)
    gt_forces.extend(np.array(gt_f).tolist())
    pred_energies_per_atom.append(pred_e / n)
    pred_forces.extend(np.array(pred_f).tolist())
    gt_energies.append(gt_e)
    pred_energies.append(pred_e)

e_mae = torch.nn.L1Loss()(torch.tensor(gt_energies_per_atom),
                          torch.tensor(pred_energies_per_atom))
f_mae = torch.nn.L1Loss()(torch.tensor(gt_forces), torch.tensor(pred_forces))
e_rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(
    gt_energies_per_atom), torch.tensor(pred_energies_per_atom)))
f_rmse = torch.sqrt(torch.nn.MSELoss()(
    torch.tensor(gt_forces), torch.tensor(pred_forces)))
rich.print(f"Energy MAE: {e_mae} eV/atom")
rich.print(f"Forces MAE: {f_mae} eV/Ang")
rich.print(f"Energy RMSE: {e_rmse} eV/atom")
rich.print(f"Forces RMSE: {f_rmse} eV/Ang")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(gt_energies, pred_energies)
plt.xlabel("True Energy (eV/structure)")
plt.ylabel("Predicted Energy (eV/structure)")
plt.title("Energy MAE: {e_mae} eV/structure")
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes,
         linestyle="-", color="k", alpha=0.7)
plt.subplot(1, 2, 2)
plt.scatter(gt_forces, pred_forces)
plt.xlabel("True Forces (eV/Ang)")
plt.ylabel("Predicted Forces (eV/Ang)")
plt.title("Forces MAE: {f_mae} eV/Ang")
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes,
         linestyle="-", color="k", alpha=0.7)
plt.tight_layout()
plt.savefig("./results/ft_parity_plot.png")
plt.close()
