import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

gOO_benchmark_data = pd.read_csv("./results/g_OO(r)-SkinnerBenmore2014.csv")
gOO_benchmark_r_values = gOO_benchmark_data["r_values"]
gOO_benchmark_g_values = gOO_benchmark_data["295.1K-g_oo"]
r_max = 6.0
indices = gOO_benchmark_r_values <= r_max
gOO_benchmark_r_values = gOO_benchmark_r_values[indices]
gOO_benchmark_g_values = gOO_benchmark_g_values[indices]

gOO_mattersim_data = np.load("./results/mattersim-1m-30-refill-g_OO(r).npz")
gOO_mattersim_r_values = gOO_mattersim_data["rdf_x"]
gOO_mattersim_g_values = gOO_mattersim_data["rdf_y"]

gOO_jmp_data = np.load("./results/jmp-s-30-refill-g_OO(r).npz")
gOO_jmp_r_values = gOO_jmp_data["rdf_x"]
gOO_jmp_g_values = gOO_jmp_data["rdf_y"]

gOO_orb_data = np.load("./results/orb-v2-30-refill-g_OO(r).npz")
gOO_orb_r_values = gOO_orb_data["rdf_x"]
gOO_orb_g_values = gOO_orb_data["rdf_y"]

gOO_eqv2_data = np.load("./results/eqv2-30-refill-g_OO(r).npz")
gOO_eqv2_r_values = gOO_eqv2_data["rdf_x"]
gOO_eqv2_g_values = gOO_eqv2_data["rdf_y"]

gOO_mace_medium_data = np.load("./results/mace_medium-30-refill-g_OO(r).npz")
gOO_mace_medium_r_values = gOO_mace_medium_data["rdf_x"]
gOO_mace_medium_g_values = gOO_mace_medium_data["rdf_y"]

gOO_mattersim_mpx2_data = np.load("./results/mattersim-1m-mpx2-g_OO(r).npz")
gOO_mattersim_mpx2_r_values = gOO_mattersim_mpx2_data["rdf_x"]
gOO_mattersim_mpx2_g_values = gOO_mattersim_mpx2_data["rdf_y"]

gOO_orbv3_data = np.load("./results/orbv3_con_omat-refill-g_OO(r).npz")
gOO_orbv3_r_values = gOO_orbv3_data["rdf_x"]
gOO_orbv3_g_values = gOO_orbv3_data["rdf_y"]

gOO_mace_medium_data = np.load("./results/mace_medium-30-refill-g_OO(r).npz")
gOO_mace_medium_r_values = gOO_mace_medium_data["rdf_x"]
gOO_mace_medium_r_values = gOO_mace_medium_r_values - 0.03
gOO_mace_medium_g_values = gOO_mace_medium_data["rdf_y"]


colors = [
    "#274753",
    "#297270",
    "#299d8f",
    "#8ab07c",
    "#e7c66b",
    "#f3a361",
]

colors.reverse()

def compute_rmse(rdf_x_true, rdf_y_true, rdf_x_pred, rdf_y_pred):
    common_true = []
    common_pred = []
    idx_true = 0
    idx_pred = 0
    while idx_true < len(rdf_x_true) and idx_pred < len(rdf_x_pred):
        if rdf_x_true[idx_true] < rdf_x_pred[idx_pred]:
            idx_true += 1
        elif rdf_x_true[idx_true] > rdf_x_pred[idx_pred]:
            idx_pred += 1
        else:
            common_true.append(rdf_y_true[idx_true])
            common_pred.append(rdf_y_pred[idx_pred])
            idx_true += 1
            idx_pred += 1
    common_true = np.array(common_true)
    common_pred = np.array(common_pred)
    rmse = np.sqrt(np.mean((common_true - common_pred) ** 2))
    return rmse

plt.figure(figsize=(8,6))
plt.scatter(gOO_benchmark_r_values, gOO_benchmark_g_values, label="Experiment", color="black", marker="o", s=10)
# plt.plot(gOO_mattersim_r_values, gOO_mattersim_g_values, color=colors[0], linewidth=2, label=f"MatterSim-V1-1M (rmse={compute_rmse(gOO_benchmark_r_values, gOO_benchmark_g_values, gOO_mattersim_r_values, gOO_mattersim_g_values):.3f})")
plt.plot(gOO_mace_medium_r_values, gOO_mace_medium_g_values, color=colors[1], linestyle="--", linewidth=2, label=f"MACE-MP-0a-medium (rmse={compute_rmse(gOO_benchmark_r_values, gOO_benchmark_g_values, gOO_mace_medium_r_values, gOO_mace_medium_g_values):.3f})")
plt.plot(gOO_eqv2_r_values, gOO_eqv2_g_values, color=colors[2], linestyle="-.", linewidth=2, label=f"EqV2-31M-mp (rmse={compute_rmse(gOO_benchmark_r_values, gOO_benchmark_g_values, gOO_eqv2_r_values, gOO_eqv2_g_values):.3f})")
plt.plot(gOO_orbv3_r_values, gOO_orbv3_g_values, color=colors[3], linestyle="dashed", linewidth=2, label=f"ORB-V3-Omat-Conserv (rmse={compute_rmse(gOO_benchmark_r_values, gOO_benchmark_g_values, gOO_orbv3_r_values, gOO_orbv3_g_values):.3f})")
# plt.plot(gOO_orb_r_values, gOO_orb_g_values, color=colors[4], linestyle="dashed", linewidth=2, label=f"ORB-V2 (rmse={compute_rmse(gOO_benchmark_r_values, gOO_benchmark_g_values, gOO_orb_r_values, gOO_orb_g_values):.3f})")
plt.plot(gOO_jmp_r_values, gOO_jmp_g_values, color=colors[5], linestyle=":", linewidth=2, label=f"JMP-S (rmse={compute_rmse(gOO_benchmark_r_values, gOO_benchmark_g_values, gOO_jmp_r_values, gOO_jmp_g_values):.3f})")

# -------- 字体大小参数 --------
AXES_LABEL_FONTSIZE = 16   # x, y 轴标题
TICK_LABEL_FONTSIZE = 14   # 刻度数字
LEGEND_FONTSIZE     = 18   # 图例文字
# -----------------------------

plt.xlabel(r"$r$ ($\AA$)", fontsize=AXES_LABEL_FONTSIZE)
plt.ylabel(r"$g_{OO}(r)$", fontsize=AXES_LABEL_FONTSIZE)

plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

plt.xlim(0, r_max)
plt.ylim(0, 5.0)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/g_OO(r)-comparison.png", dpi=300)