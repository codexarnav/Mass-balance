import pandas as pd
import numpy as np

BASE_PATH = "stress_library/01_base/stress_templates_base_60.csv"
LIT_PATH  = "stress_library/03_extracted/literature_stress_clean.csv"
OUT_PATH  = "stress_library/04_final/stress_library_final.csv"

# ------------------ Load ------------------
base = pd.read_csv(BASE_PATH)
lit  = pd.read_csv(LIT_PATH)

print("Base templates:", base.shape)
print("Literature extracted:", lit.shape)

# ------------------ Normalize literature ------------------
# Keep only expected columns
needed_cols = ["stress_type","reagent","strength","unit","pH","temperature_C","duration_hr",
               "light_lux_hr","uv_Wh_m2","humidity_RH","citation_id","paper_title","source_url"]

for c in needed_cols:
    if c not in lit.columns:
        lit[c] = np.nan

lit = lit[needed_cols]

# Add stress_id placeholder
lit["stress_id"] = ["LIT_" + str(i).zfill(4) for i in range(len(lit))]
lit["source"] = "LITERATURE"
lit["notes"] = lit["paper_title"].astype(str)

# Fill missing base-required columns
for col in base.columns:
    if col not in lit.columns:
        lit[col] = np.nan

# ------------------ Dedupe tolerance rounding ------------------
def norm_round(df):
    df = df.copy()
    df["strength_r"] = df["strength"].round(2)
    df["temperature_r"] = (df["temperature_C"] / 5).round() * 5
    df["duration_r"] = (df["duration_hr"] / 0.5).round() * 0.5
    return df

lit_r = norm_round(lit)

dedupe_key = ["stress_type","reagent","strength_r","unit","temperature_r","duration_r"]

# count occurrences
freq = lit_r.groupby(dedupe_key).size().reset_index(name="lit_count")

# keep representative first row for each dedupe key
lit_unique = lit_r.drop_duplicates(subset=dedupe_key).copy()

# merge counts back
lit_unique = lit_unique.merge(freq, on=dedupe_key, how="left")

print("Unique literature protocols after dedupe:", lit_unique.shape)

# ------------------ Compute literature weights ------------------
lit_unique["freq_weight"] = lit_unique["lit_count"] / lit_unique["lit_count"].sum()

# Scale contribution (60% base, 40% literature)
BASE_SHARE = 0.60
LIT_SHARE  = 0.40

# Normalize base weights
base = base.copy()
base["freq_weight"] = base["freq_weight"] / base["freq_weight"].sum()
base["freq_weight"] = base["freq_weight"] * BASE_SHARE

lit_unique["freq_weight"] = lit_unique["freq_weight"] * LIT_SHARE

# ------------------ Cleanup columns ------------------
# remove helper columns
lit_unique = lit_unique.drop(columns=["strength_r","temperature_r","duration_r"])

# Align column order
final_cols = list(base.columns)
final = pd.concat([base, lit_unique[final_cols]], ignore_index=True)

# Final weight normalization check
final["freq_weight"] = final["freq_weight"] / final["freq_weight"].sum()

final.to_csv(OUT_PATH, index=False)
print("✅ Final stress library saved:", OUT_PATH)
print("Final rows:", final.shape[0])
