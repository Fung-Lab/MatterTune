import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

gOO_benchmark_data = pd.read_csv("./plots/g_OO(r)-SkinnerBenmore2014.csv")
gOO_benchmark_r_values = gOO_benchmark_data["r_values"] + 0.5 * 0.06
gOO_benchmark_g_values = gOO_benchmark_data["295.1K-g_oo"]
r_max = 6.0
indices = gOO_benchmark_r_values <= r_max
gOO_benchmark_r_values = gOO_benchmark_r_values[indices]
gOO_benchmark_g_values = gOO_benchmark_g_values[indices]

gOO_cace_data = np.load("./plots/cace-g_OO(r).npz")
gOO_cace_r_values = gOO_cace_data["rdf_x"]
gOO_cace_g_values = gOO_cace_data["rdf_y"]

plt.figure(figsize=(8, 4))

plt.scatter(gOO_benchmark_r_values, gOO_benchmark_g_values, label="Experiment", color="black", marker="o", s=10)
plt.plot(gOO_cace_r_values, gOO_cace_g_values, label="CACE", color="red", linewidth=2)

plt.xlabel(r"$r$ ($\AA$)")
plt.ylabel(r"$g_{OO}(r)$")
plt.xlim(0, r_max)
plt.ylim(0, 3.0)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/g_OO(r)-comparison.png", dpi=300)