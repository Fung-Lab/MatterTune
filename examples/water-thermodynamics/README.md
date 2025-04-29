# Few-shot Ambient Water Experiment

In this folder, we show an example to fine-tune Foundation Models using only 30 samples of ambient water structures. 

To run the experiment you have to setup environment following: [installation guidance](https://fung-lab.github.io/MatterTune/installation.html)

## Quick Start

Here we give an example on how to fine-tune MatterSim model, the best-performing model so far on this experiment, with our scripts.

Firstly set up the environment:

```
git clone https://github.com/microsoft/mattersim.git
cd mattersim
conda create -n mattersim python=3.10 -y
conda activate mattersim
pip install -e .
pip install cython>=0.29.32 setuptools>=45
python setup.py build_ext --inplace
cd ..
git clone https://github.com/Fung-Lab/MatterTune.git
cd MatterTune
pip install -e .
```

Then run training scripts:

```
python water-finetune.py \
    --model_type "mattersim-1m" \
    --batch_size 16 \
    --lr 1e-4 \
    --devices 0 1 2 3 \
    --conservative 
```

You can use the fine-tuned checkpoint to run MD simulation:

```
python md.pt --ckpt_path PATH_TO_CKPT
```