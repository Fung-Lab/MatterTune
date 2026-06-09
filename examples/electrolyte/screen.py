from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read, write
from mattertune.main import load_finetuned_checkpoint
from ghost_target_calculator import GhostTargetCorrectionCalculator
from md import ASECalculatorBackedModel

from tqdm import tqdm
import matplotlib.pyplot as plt


def main(args_dict:dict):
    model = load_finetuned_checkpoint(args_dict['checkpoint_path'])
    calc = model.ase_calculator(device="cuda:0")
    calc = ASECalculatorBackedModel(
        family="uma",
        model_name=Path(args_dict['checkpoint_path']).stem,
        device="cuda:0",
        calculator=calc,
    )
    ghost_model = load_finetuned_checkpoint(args_dict['checkpoint_path'])
    ghost_calc = ghost_model.ase_calculator(device="cuda:1")
    ghost_calc = ASECalculatorBackedModel(
        family="uma",
        model_name=Path(args_dict['checkpoint_path']).stem,
        device="cuda:1",
        calculator=ghost_calc,
    )
    
    calc = GhostTargetCorrectionCalculator(
        calc,
        ghost_model=ghost_calc,
        lambda_array_name="alchemical_lambda",
        target_array_name="alchemical_target",
        epsilon=args_dict['epsilon'],
        sigma=args_dict['sigma'],
        smooth=True,
        use_d3=False,
        d3_method="pbe",
        d3_damping="d3bj"
    )
    
    atoms_list:list[Atoms] = read(args_dict['input_file'], index=":")
    time_ps_list = []
    E_lambda_list = []
    for atoms in tqdm(atoms_list):
        time_fs = atoms.info['time_fs']
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        temperature = atoms.get_temperature()
        lambda0_energy = calc.last_real_endpoint_energy
        lambda1_energy = calc.last_ghost_endpoint_energy
        E_lambda = lambda1_energy - lambda0_energy
        E_lambda_list.append(E_lambda)
        time_ps = time_fs / 1000.0
        time_ps_list.append(time_ps)
    
    plt.plot(time_ps_list, E_lambda_list)
    plt.xlabel('Time (ps)')
    plt.ylabel(r"$\delta E_{\lambda}$ (eV)")
    plt.title(r"Energy Difference $\delta E_{\lambda}$ vs Time")
    plt.tight_layout()
    plt.savefig("plot.png", dpi=300)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Screening script for electrolyte simulations")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/uma-s-1p1-best.ckpt")
    parser.add_argument("--input_file", type=str, default="./checkpoints/ghost_md.xyz")
    parser.add_argument("--epsilon", type=float, default=0.0694, help="Epsilon parameter for the correction term")
    parser.add_argument("--sigma", type=float, default=2.337, help="Sigma parameter for the correction term")
    
    args = parser.parse_args()
    main(vars(args))
    
    
