import numpy as np
import pandas as pd

# Read the data from the file
with open("./data/measured water g_OO(r) vs T - Skinner Benmore 2014.txt", "r") as f:
    data = f.readlines()
    r_values = []
    g_OO_under_temps = {
        "254.1": {
            "g_oo": [],
            "err": []
        },
        "263.1": {
            "g_oo": [],
            "err": []
        },
        "268.1": {
            "g_oo": [],
            "err": []
        },
        "277.1": {
            "g_oo": [],
            "err": []
        },
        "284.5": {
            "g_oo": [],
            "err": []
        },
        "295.1": {
            "g_oo": [],
            "err": []
        },
        "307": {
            "g_oo": [],
            "err": []
        },
        "312": {
            "g_oo": [],
            "err": []
        },
        "323.7": {
            "g_oo": [],
            "err": []
        },
        "334.1": {
            "g_oo": [],
            "err": []
        },
        "342.7": {
            "g_oo": [],
            "err": []
        },
        "354.8": {
            "g_oo": [],
            "err": []
        },
        "365.9": {
            "g_oo": [],
            "err": []
        },
    }
    for i, line in enumerate(data):
        if line.startswith("#") or line.startswith("\"#") or line.startswith("r") or line.startswith("T(K)"):
            continue
        elif i<=5:
            continue
        else:
            line_data = line.split("\t")
            if line_data[0] == "":
                continue
            r_values.append(float(line_data[0]))
            for j, temp in enumerate(g_OO_under_temps.keys()):
                g_OO_under_temps[temp]["g_oo"].append(float(line_data[2*j+1]))
                g_OO_under_temps[temp]["err"].append(float(line_data[2*j+2]))
    
# Convert the lists to numpy arrays
data = {}
data["r_values"] = np.array(r_values)
for temp in g_OO_under_temps.keys():
    data[temp+"K-g_oo"] = np.array(g_OO_under_temps[temp]["g_oo"])
    data[temp+"K-err"] = np.array(g_OO_under_temps[temp]["err"])

# Save into pandas DataFrame and save to a csv file
df = pd.DataFrame(data)
df.to_csv("./data/water_g_OO.csv", index=False)