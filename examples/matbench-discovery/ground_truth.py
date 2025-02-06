import pandas as pd
from matbench_discovery.data import DataFiles
from matbench_discovery.data import df_wbm
from matbench_discovery.enums import MbdKey

df_wbm = pd.read_csv(DataFiles.wbm_summary.path)

print(df_wbm.head())
# print all column names
print(df_wbm.columns)

mat_id_to_eform = dict(zip(df_wbm["material_id"], df_wbm[MbdKey.e_form_dft]))
for key in mat_id_to_eform.keys():
    print(key, mat_id_to_eform[key])
    break