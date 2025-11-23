from mattertune.backbones import NequIPBackboneModule
from ase.io import read
from ase import Atoms
from ase.md.langevin import Langevin
import time
import rich
from tqdm import tqdm
import copy
import wandb


atoms_pro: Atoms = read("./data/Li_electrode_test.xyz", index=0) # type: ignore
atoms_pro = atoms_pro * (4, 4, 4)  # make the system larger for better benchmarking
print(f"Number of atoms: {len(atoms_pro)}")

ckpt_path = "./checkpoints/NequIP-OAM-best.ckpt"
model = NequIPBackboneModule.load_from_checkpoint(ckpt_path)
calculator = model.ase_calculator(
    device="cuda:0"
)
atoms = copy.deepcopy(atoms_pro)
atoms.set_calculator(calculator)
dyn = Langevin(
    atoms,
    timestep=1.0,
    temperature_K=300,
    friction=0.02,
)
time1 = time.time()
for _ in tqdm(range(100)):
    dyn.step(1) # 100 steps
time2 = time.time()
rich.print(f"Inference Speed Before Compiling: {(time2 - time1)/100} seconds/step")


from nequip.ase import NequIPCalculator
calculator = NequIPCalculator.from_compiled_model(
    compile_path="./compiled_model_from_mt.nequip.pth",
    device="cuda:1",
    chemical_species_to_atom_type_map=True, 
)
atoms = copy.deepcopy(atoms_pro)
atoms.set_calculator(calculator)
dyn = Langevin(
    atoms,
    timestep=1.0,
    temperature_K=300,
    friction=0.02,
)
time1 = time.time()
for _ in tqdm(range(100)):
    dyn.step(1) # 100 steps
time2 = time.time()
rich.print(f"Inference Speed After Compiling: {(time2 - time1)/100} seconds/step")