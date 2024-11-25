from __future__ import annotations

import matbench
import matplotlib.pyplot as plt
from ase.io import read
from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False, subset=["matbench_mp_gap"])
task = list(mb.tasks)[0]
task.load()

fold = task.folds[0]
inputs_data, outputs_data = task.get_train_and_val_data(fold)

atoms_list = read("./data/gnome_Bandgap.xyz", index=":")
property_values = [atoms.info["bandgap"] for atoms in atoms_list]

plt.figure()
plt.hist(outputs_data, bins=100, label="Matbench MP Gap", alpha=0.3)
plt.hist(property_values, bins=100, label="GNoME Bandgap", alpha=0.3)
plt.xlabel("Bandgap (eV)")
plt.ylabel("Count")
plt.yscale("log")
plt.legend()
plt.savefig("./plots/bandgap_distribution.png")

plt.figure()
plt.hist(outputs_data, bins=100, label="Matbench MP Gap", alpha=0.5)
plt.xlabel("Bandgap (eV)")
plt.ylabel("Count")
plt.yscale("log")
plt.legend()
plt.savefig("./plots/matbench_mp_gap_distribution.png")

plt.figure()
plt.hist(property_values, bins=100, label="GNoME Bandgap", alpha=0.5)
plt.xlabel("Bandgap (eV)")
plt.ylabel("Count")
plt.yscale("log")
plt.legend()
plt.savefig("./plots/gnome_bandgap_distribution.png")
