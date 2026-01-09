from __future__ import annotations

import os
from ase import Atoms
from ase.io import read

from mattertune.students.allegro_model.util import allegro_model_package

ckpt_path = "./checkpoints/allegro-5.0A-T=2.ckpt"
package_path = "./checkpoints/allegro-5.0A-T=2.nequip.zip"
compiled_path = "./checkpoints/allegro-5.0A-T=2.nequip.pt2"

atoms_example:Atoms = read("./data/val_water_1593_eVAng.xyz", index=0)  # type: ignore

allegro_model_package(
    ckpt_path=ckpt_path,
    output_path=package_path,
    atoms_example=atoms_example,
)


# nequip-compile \
#   ./checkpoints/allegro-5.0A-T=2.nequip.zip \
#   ./checkpoints/allegro-5.0A-T=2.nequip.pt2 \
#   --mode aotinductor \
#   --modifiers enable_TritonContracter \
#   --device cuda \
#   --target pair_allegro