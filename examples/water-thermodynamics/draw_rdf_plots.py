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

gOO_mattersim_mpx2_data = np.load("./results/mattersim-1m-mpx2-g_OO(r).npz")
gOO_mattersim_mpx2_r_values = gOO_mattersim_mpx2_data["rdf_x"]
gOO_mattersim_mpx2_g_values = gOO_mattersim_mpx2_data["rdf_y"]

plt.scatter(gOO_benchmark_r_values, gOO_benchmark_g_values, label="Experiment", color="black", marker="o", s=10)
plt.plot(gOO_mattersim_r_values, gOO_mattersim_g_values, label="MatterSim-V1-1M (30 samples)", color="#EA8379", linewidth=2)
# plt.plot(gOO_jmp_r_values, gOO_jmp_g_values, label="JMP-S (30 samples)", color="#7DAEE0", linestyle="dashed", linewidth=2)
# plt.plot(gOO_orb_r_values, gOO_orb_g_values, label="ORB-V2 (30 samples)", color="#B395BD", linestyle=":", linewidth=2)
# plt.plot(gOO_eqv2_r_values, gOO_eqv2_g_values, label="EqV2-31M (30 samples)", color="#1B7C3D", linestyle="-.", linewidth=2)
plt.plot(gOO_mattersim_r_values, gOO_mattersim_mpx2_g_values, label="MatterSim-V1-1M-MPX2 (30 samples)", color="#7DAEE0", linestyle="dotted", linewidth=2)

plt.xlabel(r"$r$ ($\AA$)")
plt.ylabel(r"$g_{OO}(r)$")
plt.xlim(0, r_max)
plt.ylim(0, 6.0)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/g_OO(r)-comparison.png", dpi=300)