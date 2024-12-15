from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.units import Angstrom, Bohr, Hartree, eV

# 转换因子
hartree_to_eV = Hartree / eV
bohr_to_angstrom = Bohr / Angstrom
force_conversion = hartree_to_eV / bohr_to_angstrom  # Force: 1 Hartree/Bohr = eV/Å
stress_conversion = force_conversion / bohr_to_angstrom  # Stress: Hartree/Bohr³ = eV/Å³

print("Conversion factors:")
print("Hartree to eV:", hartree_to_eV)
print("Bohr to Å:", bohr_to_angstrom)
print("Force conversion factor:", force_conversion)
print("Stress conversion factor:", stress_conversion)
exit()


# 定义转换函数
def convert_to_eV_angstrom(atoms):
    """
    将原子单位制的 ASE Atoms 对象转换为 eV 和 Å 单位制。
    """
    # 转换坐标
    atoms.set_positions(atoms.get_positions() * bohr_to_angstrom)

    # 转换能量
    if "energy" in atoms.info:
        atoms.info["energy"] *= hartree_to_eV

    # 转换力
    if "forces" in atoms.arrays:
        atoms.arrays["forces"] *= force_conversion

    # 转换应力
    if "stress" in atoms.info:
        atoms.info["stress"] *= stress_conversion

    return atoms


# 示例：假设 atoms 是一个 ASE Atoms 对象，并包含原子单位制的能量、力、应力
# 创建一个简单的示例结构
atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])  # 默认单位是 Å
atoms.info["energy"] = -1.0  # Hartree
atoms.arrays["forces"] = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])  # Hartree/Bohr
atoms.info["stress"] = np.array(
    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
)  # Hartree/Bohr³ (voigt order)

# 转换单位
converted_atoms = convert_to_eV_angstrom(atoms)

# 输出结果
print("Converted positions (Å):")
print(converted_atoms.get_positions())
print("Converted energy (eV):", converted_atoms.info["energy"])
print("Converted forces (eV/Å):")
print(converted_atoms.arrays["forces"])
print("Converted stress (eV/Å³):", converted_atoms.info["stress"])
