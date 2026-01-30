import os
import pandas as pd

BASE_DIR = "stress_library"

# ---------- Create folder structure ----------
folders = [
    f"{BASE_DIR}/01_base",
    f"{BASE_DIR}/02_literature",
    f"{BASE_DIR}/02_literature/papers",
    f"{BASE_DIR}/03_extracted",
    f"{BASE_DIR}/04_final",
    f"{BASE_DIR}/scripts"
]

for f in folders:
    os.makedirs(f, exist_ok=True)

# ---------- Create placeholders ----------
# Literature template placeholder
lit_cols = [
    "stress_id","stress_type","reagent","strength","unit","pH",
    "temperature_C","duration_hr","light_lux_hr","uv_Wh_m2","humidity_RH",
    "source","freq_weight","notes",
    "citation_id","paper_title","year","authors","source_url"
]
pd.DataFrame(columns=lit_cols).to_csv(
    f"{BASE_DIR}/02_literature/stress_templates_literature.csv", index=False
)

# Scholar metadata placeholder
scholar_cols = ["paper_id","title","year","authors","venue","citation_count","pub_url","eprint_url"]
pd.DataFrame(columns=scholar_cols).to_csv(
    f"{BASE_DIR}/02_literature/scholar_results.csv", index=False
)

# Extracted raw/clean placeholders
raw_cols = [
    "paper_id","paper_title","stress_type","reagent","strength","unit",
    "temperature_C","duration_hr","pH","light_lux_hr","uv_Wh_m2","humidity_RH",
    "raw_text_snippet","source_url"
]
clean_cols = [
    "stress_type","reagent","strength","unit","pH",
    "temperature_C","duration_hr","light_lux_hr","uv_Wh_m2","humidity_RH",
    "citation_id","paper_title","year","authors","source_url"
]

pd.DataFrame(columns=raw_cols).to_csv(
    f"{BASE_DIR}/03_extracted/literature_stress_raw.csv", index=False
)
pd.DataFrame(columns=clean_cols).to_csv(
    f"{BASE_DIR}/03_extracted/literature_stress_clean.csv", index=False
)

print("✅ Step 2 complete: folders + placeholder CSVs created.")
