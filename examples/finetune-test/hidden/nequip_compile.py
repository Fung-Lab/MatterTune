from __future__ import annotations

from ase import Atoms
from ase.io import read
from mattertune.backbones.nequip_foundation.util import nequip_model_package

# # atoms: Atoms = read("../finetune-test/data/Li_electrode_test.xyz", index=0)  # type: ignore
# atoms:Atoms = read("./data/H2O.xyz", index=0)  # type: ignore
# assert len(atoms) > 3

nequip_model_package(
    ckpt_path="./checkpoints/NequIP-OAM-L-0.1-best-30-refill-conservative.ckpt",
    output_path="./checkpoints/NequIP-OAM-from-mt.nequip.zip",
)
