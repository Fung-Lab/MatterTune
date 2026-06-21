# LAMMPS MatterTune-MatterSim MD 与 Ghost-Target FEP-TI 教程

这个目录提供 MatterTune 微调后的 MatterSim 模型在 LAMMPS ML-IAP 中跑正常 MD 和
ghost-target FEP-TI MD 的示例。FEP-TI 的核心思路是：

- 从 MatterTune 训练得到的 MatterSim checkpoint 导出一个 LAMMPS 可加载的
  `.pt` ML-IAP 模型。
- 在 LAMMPS data 里把 target Li 写成一个独立 atom type，例如 type 8。
- `pair_coeff` 仍把普通 Li 和 target Li 都映射成元素 `Li`。
- 每个 lambda window 对应一个 `.pt`，因为 `lambda_value` 被写进导出的模型。

当前实现的 ghost endpoint 语义是 delete target MatterSim edges/local energy，并在
ghost endpoint 上加 target-env soft-core LJ correction。LJ correction 使用一个
MIC interaction per target-env pair，实际 cutoff 是 `min(lj_cutoff, Lmin/2)`。

## 1. 安装 Python 环境

推荐把 `mattersim` 和 `MatterTune-Elec` clone 到同一个父目录：

```bash
mkdir -p ~/workspace/electrolyte-fep
cd ~/workspace/electrolyte-fep

git clone -b electrolyte https://github.com/Lingyu-Kong/mattersim mattersim
git clone -b electrolyte https://github.com/Lingyu-Kong/MatterTune-Elec MatterTune
```

`mattersim` 的 `electrolyte` 分支要求 Python 3.12：

```bash
conda create -n mattersim-elec python=3.12 -y
conda activate mattersim-elec
export PYTHONNOUSERSITE=1

conda install -c conda-forge cmake -y
python -m pip install -U pip setuptools wheel "Cython>=0.29.32"
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install -e ./mattersim
python -m pip install -e ./MatterTune
```

如果不是 CUDA 12.6，按 PyTorch 官方安装页替换对应 CUDA index。这里必须同时从同一个
index 安装 `torch torchvision torchaudio`，否则 `torchvision` 的 C++/CUDA 算子可能和
`torch` 不匹配。

快速检查：

```bash
python -m pip check
python -c "import torch, torchvision, torchaudio, mattertune; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available())"
```

## 2. 编译安装 patched LAMMPS

MatterSim 仓库里带了 LAMMPS ML-IAP bridge patch 和构建脚本。下面的命令会 clone 官方
LAMMPS，checkout `stable` 分支，应用 patch，编译 Kokkos CUDA + ML-IAP，并把 `lmp`、
`liblammps.so` 和 LAMMPS Python wheel 安装进当前 conda 环境。

```bash
cd ~/workspace/electrolyte-fep/mattersim

bash scripts/build_lammps_mliap_kokkos.sh \
  --work-root ~/workspace/electrolyte-fep/lammps-work \
  --ref stable \
  --cuda-root /usr/local/cuda-12.6 \
  --kokkos-arch AMPERE86
```

`--work-root` 是构建工作目录，脚本会在里面创建 `lammps/` source 和 build 目录。如果你已经
手动 clone 了 LAMMPS，可以改用：

```bash
bash scripts/build_lammps_mliap_kokkos.sh \
  --source-dir /path/to/lammps \
  --build-dir /path/to/lammps/build-mattersim-mliap-kokkos \
  --skip-clone \
  --ref stable \
  --cuda-root /usr/local/cuda-12.6 \
  --kokkos-arch AMPERE86
```

Kokkos architecture 常用值：

| GPU | `--kokkos-arch` |
| --- | --- |
| A100 | `AMPERE80` |
| RTX A6000/A5000/A4000 | `AMPERE86` |
| H100/H200 | `HOPPER90` |
| L40/L40S/RTX 4090 | `ADA89` |

编译后检查：

```bash
${CONDA_PREFIX}/bin/lmp -h | grep -E "Git info|Installed packages|KOKKOS package API"
python -c "import lammps; from mattersim.lammps.ghost_target_mliap_wrapper import GhostTargetMatterSimMLIAP; print(lammps.__file__, GhostTargetMatterSimMLIAP.__name__)"
```

成功的 LAMMPS 应该显示包含 `KOKKOS ML-IAP ML-SNAP PYTHON`。

## 3. 方法一：手动导出 `.pt`、准备 data 和手写 `in.lammps`

这种方法适合你想完全控制每个 lambda window 的输入文件。它不调用
`run_lammps_fep_ti.sh`。

### 3.1 设置路径和参数

```bash
cd ~/workspace/electrolyte-fep/MatterTune
conda activate mattersim-elec
export PYTHONNOUSERSITE=1
export PYTHONPATH=~/workspace/electrolyte-fep/mattersim/src:~/workspace/electrolyte-fep/MatterTune/src:${PYTHONPATH:-}

CKPT=/path/to/mattersim-best.ckpt
PDB=/path/to/top.pdb
RUN_DIR=/path/to/run_lambda050
LAMBDA_VALUE=0.5
TARGET_TYPE=8
mkdir -p "${RUN_DIR}"
```

`TARGET_TYPE` 是 LAMMPS atom type，不是元素序号。对当前 `ELEMENT_ORDER=Li,F,S,N,O,C,H`，
普通元素 type 是 1 到 7，因此 target Li 可以用 type 8。

### 3.2 从 checkpoint 导出 LAMMPS ML-IAP `.pt`

```bash
python examples/elec-Li-new/enhance-V1/export_mattertune_ghost_target_lammps_mliap.py \
  --checkpoint "${CKPT}" \
  --output "${RUN_DIR}/ghost-target-lambda050-type8-rc10.0.pt" \
  --lambda-value "${LAMBDA_VALUE}" \
  --target-types "${TARGET_TYPE}" \
  --epsilon 0.00694 \
  --sigma 2.337 \
  --lj-cutoff 10.0 \
  --energy-log-path "${RUN_DIR}/fep-ti-energy.raw.csv" \
  --energy-log-interval 1 \
  --energy-log-timestep-fs 1.0 \
  --device cpu
```

这个 `.pt` 内部保存了：

- MatterTune checkpoint unwrap 得到的 MatterSim Potential。
- MatterTune energy denormalizer/reference 信息。
- `lambda_value`。
- `target_types`，也就是哪些 LAMMPS atom type 被当成 ghost target。
- soft-core LJ 参数。
- FEP-TI energy log 的输出路径、输出频率和 timestep。

所以不同 lambda window 要导出不同 `.pt`，例如 lambda 0.0、0.5、1.0 各一个。
LAMMPS 运行时会由 ML-IAP wrapper 写出 `${RUN_DIR}/fep-ti-energy.raw.csv`，每次 force
evaluation append 一行 CSV，并立刻 flush。这个 raw 文件包含 endpoint energy，但
`temperature_K` 只是占位；最终给后处理使用的是合并后的 `${RUN_DIR}/fep-ti-energy.log`。
字段和 ASE `examples/electrolyte/md.py` 的 energy log 保持一致：

```text
step,time_fs,time_ps,temperature_K,mixed_energy_eV,E_I_eV,E_F_with_LJ_eV,E_F_without_LJ_eV,E_LJ_eV,deltaE_with_LJ_eV,deltaE_without_LJ_eV
```

其中 `E_I_eV` 是 real endpoint，`E_F_without_LJ_eV` 是 delete ghost MatterSim base endpoint，
`E_LJ_eV` 是 ghost endpoint 的 LJ correction，`E_F_with_LJ_eV` 是 ghost base + LJ，
`mixed_energy_eV` 是 LAMMPS 实际使用的 lambda 插值能量。`temperature_K` 来自 LAMMPS
`fix print ... screen no` 写出的 sidecar，再由 merge 脚本填回最终 CSV。

注意这个日志路径保存在导出的 `.pt` 里。如果你之前已经导出过旧 `.pt`，需要重新导出；使用
`run_lammps_fep_ti.sh` 时可以加 `--force-export`。通常每个 MD step 对应一次 force evaluation；
LAMMPS 在 setup、`run 0` 或 warmup/production 切换时如果额外评估一次力，CSV 也会忠实多写一行。

### 3.3 准备 LAMMPS data

你可以自己写 data 文件，只要满足下面约定：

- `atom_style atomic`
- target Li 的 atom type 使用 `TARGET_TYPE=8`
- 普通 Li 仍是 type 1
- `Masses` 里 type 8 的质量仍写 Li 质量
- `pair_coeff * *` 最后一项仍映射成 `Li`

例如 data 文件的关键部分应该类似：

```text
8 atom types

Masses

1 6.94 # Li
2 18.998403163 # F
3 32.06 # S
4 14.007 # N
5 15.999 # O
6 12.011 # C
7 1.008 # H
8 6.94 # Li

Atoms # atomic

1 8  x0 y0 z0  # target Li, PDB zero-based index 0
2 1  x1 y1 z1  # ordinary Li
...
```

如果你只有 PDB，也可以用本目录的 helper 先生成 data 和一个可参考的 input，然后自己修改
`in.lammps`：

```bash
python examples/elec-Li-new/mix_further_ft/prepare_lammps_ghost_target_input.py \
  --pdb "${PDB}" \
  --data "${RUN_DIR}/top_target_type8.data" \
  --input "${RUN_DIR}/in.lambda050.template.lammps" \
  --model "${RUN_DIR}/ghost-target-lambda050-type8-rc10.0.pt" \
  --metadata "${RUN_DIR}/prepare_metadata.json" \
  --target-indices 0 \
  --target-type "${TARGET_TYPE}" \
  --element-order Li,F,S,N,O,C,H \
  --temperature 298.15 \
  --timestep-fs 1.0 \
  --friction-fs-inv 0.02 \
  --warmup-steps 20 \
  --steps 100000 \
  --thermo-interval 100 \
  --dump-interval 100 \
  --seed 7 \
  --final-data "${RUN_DIR}/final_lambda050.data" \
  --dump "${RUN_DIR}/traj_lambda050.lammpstrj" \
  --temperature-log "${RUN_DIR}/fep-ti-temperature.csv" \
  --temperature-log-interval 1
```

这里的 `--target-indices` 是 PDB/ASE 的 zero-based atom index。比如 `0` 表示 PDB 里第一个原子。

### 3.4 手写 `in.lammps`

下面是一个最小 NVT Langevin 模板。注意 `pair_coeff` 有 8 个元素映射，最后一个 `Li` 对应
target Li 的 atom type 8。

```lammps
units           metal
atom_style      atomic
boundary        p p p

variable        DATA_PATH string /path/to/run_lambda050/top_target_type8.data
variable        MODEL_PATH string /path/to/run_lambda050/ghost-target-lambda050-type8-rc10.0.pt

newton          on
read_data       ${DATA_PATH}

pair_style      mliap unified ${MODEL_PATH}
pair_coeff      * * Li F S N O C H Li

velocity        all create 298.15 7 mom yes rot yes dist gaussian
timestep        0.001
fix             int all nve
fix             therm all langevin 298.15 298.15 0.05 7926 zero yes

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

thermo          100
thermo_style    custom step temp pe ke etotal press
thermo_modify   flush yes

run             20
reset_timestep  0

dump            traj all custom 100 /path/to/run_lambda050/traj_lambda050.lammpstrj id type element x y z vx vy vz fx fy fz
dump_modify     traj element Li F S N O C H Li sort id
run             100000
undump          traj

write_data      /path/to/run_lambda050/final_lambda050.data
```

这里 `friction_fs_inv=0.02 fs^-1` 对应 Langevin damping time：

```text
damping_ps = (1 / 0.02 fs) * 0.001 ps/fs = 0.05 ps
```

### 3.5 运行 LAMMPS

```bash
PYTHONNOUSERSITE=1 ${CONDA_PREFIX}/bin/lmp \
  -k on g 1 \
  -sf kk \
  -pk kokkos newton on neigh half \
  -in "${RUN_DIR}/in.lambda050.lammps" \
  -log "${RUN_DIR}/log.lambda050.lammps"
```

如果手动跑而不是用 `run_lammps_fep_ti.sh`，LAMMPS 完成后再合并正式 energy log：

```bash
python examples/elec-Li-new/mix_further_ft/merge_lammps_fep_ti_energy_log.py \
  --energy-log-raw "${RUN_DIR}/fep-ti-energy.raw.csv" \
  --temperature-log "${RUN_DIR}/fep-ti-temperature.csv" \
  --output "${RUN_DIR}/fep-ti-energy.log"
```

`pair_mliap` 需要 `newton on`，Kokkos 路径建议使用 half neighbor list：
`-pk kokkos newton on neigh half`。

## 4. 方法二：使用 `run_lammps_fep_ti.sh`

`run_lammps_fep_ti.sh` 是把方法一自动化。它没有引入新的物理模型，主要自动完成：

1. 激活 conda 环境，设置 `PYTHONNOUSERSITE=1` 和 `PYTHONPATH`。
2. 根据 `lambda_value` 和时间戳创建 `RUN_DIR`。
3. 调用 `export_mattertune_ghost_target_lammps_mliap.py`，从 checkpoint 导出 `.pt`，并把
   raw energy CSV 的输出路径写进 `.pt`。
4. 调用 `prepare_lammps_ghost_target_input.py`，从 PDB 写出 LAMMPS data、`in.lammps`、metadata
   和每步 temperature sidecar 配置。
5. 调用 `${LMP_BIN}` 用 Kokkos/ML-IAP 跑 warmup 和 production MD。
6. 调用 `merge_lammps_fep_ti_energy_log.py`，把 raw energy 和 temperature sidecar 合并成
   `fep-ti-energy.log`。
7. 保存 log、trajectory、final data 和本次运行参数。

最常用命令：

```bash
cd ~/workspace/electrolyte-fep/MatterTune

bash examples/elec-Li-new/mix_further_ft/run_lammps_fep_ti.sh \
  --lambda-value 0.5 \
  --checkpoint /path/to/mattersim-best.ckpt \
  --structure /path/to/top.pdb \
  --target-indices 0 \
  --cuda-visible-devices 0
```

常用覆盖项：

```bash
bash examples/elec-Li-new/mix_further_ft/run_lammps_fep_ti.sh \
  --lambda-value 0.5 \
  --checkpoint /path/to/mattersim-best.ckpt \
  --structure /path/to/top.pdb \
  --run-dir /path/to/run_lambda050 \
  --steps 100000 \
  --warmup-steps 20 \
  --temperature 298.15 \
  --timestep-fs 1.0 \
  --friction-fs-inv 0.02 \
  --energy-log-interval 1 \
  --target-indices 0 \
  --lj-cutoff 10.0 \
  --cuda-visible-devices 0
```

只准备 `.pt`、data 和 `in.lammps`，不启动 MD：

```bash
bash examples/elec-Li-new/mix_further_ft/run_lammps_fep_ti.sh \
  --lambda-value 0.5 \
  --checkpoint /path/to/mattersim-best.ckpt \
  --structure /path/to/top.pdb \
  --target-indices 0 \
  --prepare-only
```

这会生成：

```text
ghost-target-lambdaXXX-type8-rc10.0.pt
top_target_type8.data
in.lambdaXXX.lammps
prepare_metadata.json
run_lammps_fep_ti_config.txt
```

如果不加 `--prepare-only`，还会生成：

```text
fep-ti-energy.log
fep-ti-energy.raw.csv
fep-ti-temperature.csv
log.lambdaXXX.lammps
traj_lambdaXXX.lammpstrj
final_lambdaXXX.data
```

## 5. 方法三：正常 MatterTune-MatterSim LAMMPS MD（非 FEP-TI）

如果只想用微调后的 MatterSim checkpoint 在 LAMMPS 里跑正常 MD，不需要 ghost target、
lambda、soft-core LJ，也不需要把 target Li 写成独立 atom type。正常 MD 使用普通
`MatterTuneMatterSimMLIAP` 导出的 `.pt`：

- data 文件只需要真实元素 atom type，例如 `Li,F,S,N,O,C,H` 共 7 个 type。
- `pair_coeff * *` 后面的元素数量等于 atom type 数量。
- 对当前体系，所有 Li 都是 type 1，没有 type 8。
- 导出的 `.pt` 会保留 MatterTune energy denormalizer/reference，所以 LAMMPS thermo 里的
  `pe` 是微调模型的完整 energy。

不要用 `prepare_lammps_ghost_target_input.py` 或 `run_lammps_fep_ti.sh` 准备正常 MD；
它们是 ghost-target FEP-TI 专用，会引入 target type 和 lambda 模型。

### 5.1 从 checkpoint 导出正常 MD 的 `.pt`

```bash
cd ~/workspace/electrolyte-fep/MatterTune
conda activate mattersim-elec
export PYTHONNOUSERSITE=1
export PYTHONPATH=~/workspace/electrolyte-fep/mattersim/src:~/workspace/electrolyte-fep/MatterTune/src:${PYTHONPATH:-}

export CKPT=/path/to/mattersim-best.ckpt
export PDB=/path/to/top.pdb
export RUN_DIR=/path/to/normal_md_run
mkdir -p "${RUN_DIR}"

python examples/elec-Li-new/enhance-V1/export_mattersim_lammps_mliap.py \
  "${CKPT}" \
  "${RUN_DIR}/mattertune-mattersim-normal.pt" \
  --device cpu
```

如果运行时不想使用 `torch.compile`，导出时加 `--no-compile`：

```bash
python examples/elec-Li-new/enhance-V1/export_mattersim_lammps_mliap.py \
  "${CKPT}" \
  "${RUN_DIR}/mattertune-mattersim-normal-no-compile.pt" \
  --device cpu \
  --no-compile
```

### 5.2 从 PDB 写正常 MD 的 LAMMPS data

下面这个最小 writer 只支持当前 electrolyte PDB 常见的 orthorhombic cell，并按
`Li,F,S,N,O,C,H` 写 7 个 atom type。它不会创建 ghost target type。

```bash
python - <<'PY'
from pathlib import Path
import math
import os

pdb = Path(os.environ["PDB"])
out = Path(os.environ["RUN_DIR"]) / "top_normal.data"

element_order = ("Li", "F", "S", "N", "O", "C", "H")
masses = {
    "Li": 6.94,
    "F": 18.998403163,
    "S": 32.06,
    "N": 14.007,
    "O": 15.999,
    "C": 12.011,
    "H": 1.008,
}
types = {element: i + 1 for i, element in enumerate(element_order)}


def element_from_name(name):
    letters = "".join(ch for ch in name.strip() if ch.isalpha())
    if not letters:
        raise ValueError(f"Cannot infer element from atom name {name!r}")
    upper = letters.upper()
    if upper.startswith("LI"):
        return "Li"
    element = upper[0]
    if element not in types:
        raise ValueError(f"Unsupported element inferred from {name!r}: {element}")
    return element


def wrap(value, length):
    value = math.fmod(value, length)
    return value + length if value < 0.0 else value


cell = None
atoms = []
with pdb.open(encoding="utf-8") as handle:
    for line in handle:
        record = line[:6].strip()
        if record == "CRYST1":
            fields = line.split()
            a, b, c = (float(fields[i]) for i in (1, 2, 3))
            alpha, beta, gamma = (float(fields[i]) for i in (4, 5, 6))
            if any(abs(angle - 90.0) > 1e-6 for angle in (alpha, beta, gamma)):
                raise ValueError("This simple writer only supports orthorhombic cells.")
            cell = (a, b, c)
        elif record in {"ATOM", "HETATM"}:
            name = line[12:16].strip()
            atoms.append(
                (
                    len(atoms) + 1,
                    element_from_name(name),
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )

if cell is None:
    raise ValueError(f"No CRYST1 record found in {pdb}")
if not atoms:
    raise ValueError(f"No atoms found in {pdb}")

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    handle.write("LAMMPS data generated from PDB for normal MatterSim MD\n\n")
    handle.write(f"{len(atoms)} atoms\n")
    handle.write(f"{len(element_order)} atom types\n\n")
    handle.write(f"0.0 {cell[0]:.10f} xlo xhi\n")
    handle.write(f"0.0 {cell[1]:.10f} ylo yhi\n")
    handle.write(f"0.0 {cell[2]:.10f} zlo zhi\n\n")
    handle.write("Masses\n\n")
    for atom_type, element in enumerate(element_order, start=1):
        handle.write(f"{atom_type} {masses[element]:.12g} # {element}\n")
    handle.write("\nAtoms # atomic\n\n")
    for atom_id, element, x, y, z in atoms:
        handle.write(
            f"{atom_id} {types[element]} "
            f"{wrap(x, cell[0]):.10f} {wrap(y, cell[1]):.10f} {wrap(z, cell[2]):.10f}\n"
        )

print(f"wrote {out}")
print("pair_coeff * * " + " ".join(element_order))
PY
```

如果你使用自己的 data writer，只需要保证 atom type 和 `pair_coeff` 的元素顺序一致。例如 type
1 是 Li、type 2 是 F，就写 `pair_coeff * * Li F ...`。

### 5.3 手写正常 MD 的 `in.lammps`

写入 `${RUN_DIR}/in.normal.lammps`：

```lammps
units           metal
atom_style      atomic
boundary        p p p

variable        DATA_PATH string /path/to/normal_md_run/top_normal.data
variable        MODEL_PATH string /path/to/normal_md_run/mattertune-mattersim-normal.pt

newton          on
read_data       ${DATA_PATH}

pair_style      mliap unified ${MODEL_PATH}
pair_coeff      * * Li F S N O C H

velocity        all create 298.15 7 mom yes rot yes dist gaussian
timestep        0.001
fix             int all nve
fix             therm all langevin 298.15 298.15 0.05 7926 zero yes

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

thermo          100
thermo_style    custom step temp pe ke etotal press
thermo_modify   flush yes

run             20
reset_timestep  0

dump            traj all custom 100 /path/to/normal_md_run/traj_normal.lammpstrj id type element x y z vx vy vz fx fy fz
dump_modify     traj element Li F S N O C H sort id
run             100000
undump          traj

write_data      /path/to/normal_md_run/final_normal.data
```

这里仍然是 `units metal`，所以 `timestep 0.001` 是 1 fs，Langevin damping `0.05` 是 0.05 ps，
对应 `friction_fs_inv=0.02 fs^-1`。

### 5.4 运行正常 MD

```bash
PYTHONNOUSERSITE=1 ${CONDA_PREFIX}/bin/lmp \
  -k on g 1 \
  -sf kk \
  -pk kokkos newton on neigh half \
  -in "${RUN_DIR}/in.normal.lammps" \
  -log "${RUN_DIR}/log.normal.lammps"
```

建议第一次先把 input 里的 production `run 100000` 改成 `run 0` 或 `run 100`，确认
`pe`、forces 和 neighbor list 都正常，再跑长 MD。

### 5.5 使用 `run_lammps_mlp_md.sh` 一键跑正常 MD

如果不想手动导出 `.pt` 和写 `in.lammps`，可以直接用本目录的一键脚本。它做的事情是：

1. 激活 conda 环境，设置 `PYTHONNOUSERSITE=1` 和 `PYTHONPATH`。
2. 调用 `export_mattersim_lammps_mliap.py`，从 MatterTune checkpoint 导出普通 ML-IAP `.pt`。
3. 调用 `prepare_lammps_normal_input.py`，从 PDB 写出正常 LAMMPS data、`in.normal.lammps` 和 metadata。
4. 调用 `${LMP_BIN}` 用 Kokkos/ML-IAP 跑 warmup 和 production MD。
5. 保存 log、trajectory、final data 和本次运行参数。

最常用命令：

```bash
cd ~/workspace/electrolyte-fep/MatterTune

bash examples/elec-Li-new/mix_further_ft/run_lammps_mlp_md.sh \
  --checkpoint /path/to/mattersim-best.ckpt \
  --structure /path/to/top.pdb \
  --cuda-visible-devices 0
```

常用覆盖项：

```bash
bash examples/elec-Li-new/mix_further_ft/run_lammps_mlp_md.sh \
  --checkpoint /path/to/mattersim-best.ckpt \
  --structure /path/to/top.pdb \
  --run-dir /path/to/normal_md_run \
  --steps 100000 \
  --warmup-steps 20 \
  --temperature 298.15 \
  --timestep-fs 1.0 \
  --friction-fs-inv 0.02 \
  --thermo-interval 100 \
  --dump-interval 100 \
  --cuda-visible-devices 0
```

只准备 `.pt`、data 和 `in.normal.lammps`，不启动 MD：

```bash
bash examples/elec-Li-new/mix_further_ft/run_lammps_mlp_md.sh \
  --checkpoint /path/to/mattersim-best.ckpt \
  --structure /path/to/top.pdb \
  --prepare-only
```

这会生成：

```text
mattertune-mattersim-normal.pt
top_normal.data
in.normal.lammps
prepare_metadata.json
run_lammps_mlp_md_config.txt
```

如果不加 `--prepare-only`，还会生成：

```text
log.normal.lammps
traj_normal.lammpstrj
final_normal.data
```

和 FEP-TI 脚本不同，`run_lammps_mlp_md.sh` 不会创建 target atom type，也不会写
`fep-ti-energy.log`；正常 MD 的能量看 LAMMPS `log.normal.lammps` 里的 thermo `pe`。

## 6. 跑多个 lambda windows

一个简单 loop：

```bash
for lam in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  bash examples/elec-Li-new/mix_further_ft/run_lammps_fep_ti.sh \
    --lambda-value "${lam}" \
    --checkpoint /path/to/mattersim-best.ckpt \
    --structure /path/to/top.pdb \
    --target-indices 0 \
    --cuda-visible-devices 0
done
```

每个 window 会有自己的 `.pt`、`in.lammps` 和运行目录。后处理时从每个 lambda window 的轨迹或
log 中提取相应的 FEP-TI 统计量。

## 7. 常见问题

- `Package 'mattersim' requires a different Python`: 使用 `python=3.12`。
- `torchvision::nms does not exist`: 用同一个 PyTorch CUDA index 同时安装
  `torch torchvision torchaudio`。
- `cmake: command not found`: 在目标 conda 环境里安装 `cmake`。
- `Could NOT find Cythonize`: 在目标 conda 环境里安装 `Cython`。
- `No module named 'lammps'`: LAMMPS Python wheel 没装。新版 build helper 默认会运行
  `cmake --build ... --target install-python`；旧 build 需要手动运行这个 target。
- `pair_coeff` 报元素数量不匹配：LAMMPS atom type 数量要和 `pair_coeff * *` 后面的元素数一致。
  如果 target type 是 8，就需要 `Li F S N O C H Li` 这 8 个映射。
- `target_types` 没生效：确认 data 文件里 target atom 的 LAMMPS atom type 确实是 `--target-type`
  指定的值，而不是普通 Li 的 type 1。
