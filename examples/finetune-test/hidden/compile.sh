#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

conda activate nequip-tune

nequip-compile \
  ./checkpoints/NequIP-OAM-from-mt.nequip.zip \
  ./checkpoints/compiled_model_from_mt.nequip.pth \
  --mode torchscript \
  --device cuda \
  --target ase 

nequip-compile \
  /nethome/lkong88/workspace/nequip/examples/NequIP-OAM-L-0.1.nequip.zip \
  ./checkpoints/mir-group__NequIP-OAM-L__0.1.nequip.pth \
  --mode torchscript \
  --device cuda \
  --target ase
  
nequip-compile \
  ./checkpoints/NequIP-OAM-from-mt.nequip.zip \
  ./checkpoints/compiled_model_from_mt-aotinductor.nequip.pt2 \
  --mode aotinductor \
  --device cuda \
  --target ase 

nequip-compile \
  /nethome/lkong88/workspace/nequip/examples/NequIP-OAM-L-0.1.nequip.zip \
  ./checkpoints/mir-group__NequIP-OAM-L__0.1-aotinductor.nequip.pt2 \
  --mode aotinductor \
  --device cuda \
  --target ase