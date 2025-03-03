import os
import rich

import numpy as np
from ase.io import read
from matbench_discovery.metrics.discovery import stable_metrics

PATH = "/net/csefiles/coc-fung-cluster/lingyu/matbench-discovery"
MODEL = "eqv2"

dir_path = os.path.join(PATH, MODEL)

xyz_files = os.listdir(dir_path)

atoms_list = []
for file in xyz_files:
    atoms_list.extend(read(os.path.join(dir_path, file), ":"))
    
print(len(atoms_list))
assert len(atoms_list) == 256963

e_hull_preds = np.array([atoms.info["e_hull_pred"] for atoms in atoms_list])
e_hull_trues = np.array([atoms.info["e_hull_true"] for atoms in atoms_list])

rich.print(stable_metrics(e_hull_trues, e_hull_preds))