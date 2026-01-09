from ase import Atoms
from ase.io import read

atoms_list:list[Atoms] = read("/net/csefiles/coc-fung-cluster/lingyu/datasets/li3po4-train.xyz", ":")  # type: ignore
print(f"Loaded {len(atoms_list)} atoms from the dataset.")