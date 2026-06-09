from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "electrolyte"))
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "electrolyte_md_verify", ROOT / "examples" / "electrolyte" / "md.py"
)
assert spec is not None and spec.loader is not None
md = importlib.util.module_from_spec(spec)
sys.modules["electrolyte_md_verify"] = md
spec.loader.exec_module(md)

from ghost_target_calculator import GhostTargetCorrectionCalculator  # noqa: E402


def main() -> None:
    ckpt = (
        "/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/"
        "03-train-pair-100/20260524-225554-uma-uma-s1.1-pair100-fw20-de05/"
        "checkpoints/uma-uma-s-1p1-pair100-best.ckpt"
    )
    structure = (
        "/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/get_all_trj_aimd/"
        "Li-metal/case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/top.pdb"
    )
    device = "cuda:7"
    torch.cuda.set_device(7)
    model = md.load_finetuned_model_from_checkpoint(ckpt, device=device)
    atoms = read(structure)
    target = np.zeros(len(atoms), dtype=bool)
    target[0] = True
    lam = np.zeros(len(atoms), dtype=np.float64)
    lam[0] = 0.0
    atoms.arrays["alchemical_target"] = target
    atoms.arrays["alchemical_lambda"] = lam
    calc = GhostTargetCorrectionCalculator(
        model,
        lambda_array_name="alchemical_lambda",
        target_array_name="alchemical_target",
        epsilon=0.00694,
        sigma=2.337,
    )
    atoms.calc = calc

    counter = {"n": 0}
    original_calculate = calc.calculate

    def wrapped_calculate(*args, **kwargs):
        counter["n"] += 1
        return original_calculate(*args, **kwargs)

    calc.calculate = wrapped_calculate  # type: ignore[method-assign]

    rng = np.random.default_rng(0)
    atoms.positions += rng.normal(scale=1.0e-6, size=atoms.positions.shape)
    atoms.get_forces()
    atoms.get_potential_energy()
    torch.cuda.synchronize()

    counter["n"] = 0
    times = []
    for _ in range(10):
        atoms.positions += rng.normal(scale=1.0e-6, size=atoms.positions.shape)
        start = time.perf_counter()
        atoms.get_forces()
        atoms.get_potential_energy()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    print(f"calculate_calls={counter['n']}")
    print(f"cycles={len(times)}")
    print(f"calls_per_cycle={counter['n'] / len(times):.3f}")
    print(f"median_s={statistics.median(times):.6f}")
    print(f"mean_s={statistics.fmean(times):.6f}")
    print("times_s=" + ",".join(f"{value:.6f}" for value in times))


if __name__ == "__main__":
    main()
