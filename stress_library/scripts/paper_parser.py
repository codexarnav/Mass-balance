import os, re
import pandas as pd
import pdfplumber
from pathlib import Path

# Optional libraries
try:
    import camelot
except:
    camelot = None

PAPERS_DIR = "stress_library/02_literature/papers"
OUT_RAW = "stress_library/03_extracted/literature_stress_raw.csv"
OUT_CLEAN = "stress_library/03_extracted/literature_stress_clean.csv"

os.makedirs("stress_library/03_extracted", exist_ok=True)

# ---------------- REGEX PATTERNS ----------------
acid_pat = re.compile(r"(\d+(\.\d+)?)\s*(N|M)\s*(HCl|hydrochloric acid)", re.I)
base_pat = re.compile(r"(\d+(\.\d+)?)\s*(N|M)\s*(NaOH|sodium hydroxide)", re.I)
ox_pat   = re.compile(r"(\d+(\.\d+)?)\s*%\s*(H2O2|hydrogen peroxide)", re.I)

temp_pat = re.compile(r"(\d+(\.\d+)?)\s*°\s*C", re.I)
time_pat = re.compile(r"(\d+(\.\d+)?)\s*(h|hr|hrs|hour|hours|min|mins|minutes|day|days)", re.I)

rh_pat   = re.compile(r"(\d+)\s*%\s*(RH|relative humidity)", re.I)
lux_pat  = re.compile(r"(\d+(\.\d+)?)\s*(million)?\s*lux", re.I)
uv_pat   = re.compile(r"(\d+(\.\d+)?)\s*wh\s*/\s*m2", re.I)

def parse_time_to_hr(t):
    if not t:
        return None
    s = t.lower()
    num = float(re.search(r"(\d+(\.\d+)?)", s).group(1))
    if "min" in s:
        return num / 60
    if "day" in s:
        return num * 24
    return num

def safe_float(x):
    try: return float(x)
    except: return None

def extract_matches(text_block, paper_id, paper_title, source_type="TEXT"):
    """Extract stress protocol matches from a text block."""
    raw = []
    clean = []

    text_block = re.sub(r"\s+", " ", str(text_block))

    matches = []
    for m in acid_pat.finditer(text_block):
        matches.append(("acid_hydrolysis", m.group(0), m.start()))
    for m in base_pat.finditer(text_block):
        matches.append(("base_hydrolysis", m.group(0), m.start()))
    for m in ox_pat.finditer(text_block):
        matches.append(("oxidation", m.group(0), m.start()))
    for m in rh_pat.finditer(text_block):
        matches.append(("humidity", m.group(0), m.start()))
    for m in lux_pat.finditer(text_block):
        matches.append(("photolysis", m.group(0), m.start()))
    for m in uv_pat.finditer(text_block):
        matches.append(("photolysis", m.group(0), m.start()))
    for m in temp_pat.finditer(text_block):
        matches.append(("thermal", m.group(0), m.start()))

    for stype, found, pos in matches:
        snippet = text_block[max(pos-150, 0):pos+250]
        temp_m = temp_pat.search(snippet)
        time_m = time_pat.search(snippet)
        rh_m = rh_pat.search(snippet)
        lux_m = lux_pat.search(snippet)
        uv_m = uv_pat.search(snippet)

        raw.append({
            "paper_id": paper_id,
            "paper_title": paper_title,
            "source_type": source_type,
            "stress_type": stype,
            "detected_text": found,
            "near_temp": temp_m.group(0) if temp_m else None,
            "near_time": time_m.group(0) if time_m else None,
            "near_rh": rh_m.group(0) if rh_m else None,
            "near_lux": lux_m.group(0) if lux_m else None,
            "near_uv": uv_m.group(0) if uv_m else None,
            "raw_text_snippet": snippet[:400]
        })

        # Normalize
        reagent = None
        strength = None
        unit = None

        if stype == "acid_hydrolysis":
            reagent = "HCl"
            mm = re.search(r"(\d+(\.\d+)?)\s*(N|M)", found, re.I)
            if mm:
                strength = safe_float(mm.group(1))
                unit = mm.group(3).upper()

        elif stype == "base_hydrolysis":
            reagent = "NaOH"
            mm = re.search(r"(\d+(\.\d+)?)\s*(N|M)", found, re.I)
            if mm:
                strength = safe_float(mm.group(1))
                unit = mm.group(3).upper()

        elif stype == "oxidation":
            reagent = "H2O2"
            mm = re.search(r"(\d+(\.\d+)?)\s*%", found)
            if mm:
                strength = safe_float(mm.group(1))
                unit = "%"

        elif stype == "thermal":
            reagent = "dry_heat"
            unit = "NA"

        elif stype == "humidity":
            reagent = "RH"
            unit = "NA"

        elif stype == "photolysis":
            reagent = "light"
            unit = "NA"

        temperature_C = None
        if temp_m:
            temperature_C = safe_float(re.search(r"(\d+(\.\d+)?)", temp_m.group(0)).group(1))

        duration_hr = parse_time_to_hr(time_m.group(0)) if time_m else None

        humidity_RH = safe_float(rh_m.group(1)) if rh_m else None

        light_lux_hr = None
        if lux_m:
            num = safe_float(lux_m.group(1))
            if lux_m.group(3):
                light_lux_hr = num * 1_000_000
            else:
                light_lux_hr = num

        uv_Wh_m2 = safe_float(uv_m.group(1)) if uv_m else None

        clean.append({
            "stress_type": stype,
            "reagent": reagent,
            "strength": strength,
            "unit": unit,
            "temperature_C": temperature_C,
            "duration_hr": duration_hr,
            "light_lux_hr": light_lux_hr,
            "uv_Wh_m2": uv_Wh_m2,
            "humidity_RH": humidity_RH,
            "citation_id": paper_id,
            "paper_title": paper_title
        })

    return raw, clean


raw_rows, clean_rows = [], []

pdf_files = list(Path(PAPERS_DIR).glob("*.pdf"))
print(f"✅ Found {len(pdf_files)} PDFs")

for pdf in pdf_files:
    paper_id = pdf.stem.split("_")[0]
    paper_title = pdf.stem

    print(f"\n📄 Processing: {pdf.name}")

    # -------- 1) Extract TEXT + TABLES with pdfplumber ----------
    try:
        with pdfplumber.open(str(pdf)) as pdfobj:
            for i, page in enumerate(pdfobj.pages[:8]):  # first 8 pages enough
                page_text = page.extract_text() or ""
                rr, cc = extract_matches(page_text, paper_id, paper_title, source_type="TEXT")
                raw_rows.extend(rr)
                clean_rows.extend(cc)

                # tables from pdfplumber
                tables = page.extract_tables()
                for t in tables:
                    # convert table rows into a text block
                    t_text = "\n".join([" ".join([str(x) for x in row if x]) for row in t])
                    rr, cc = extract_matches(t_text, paper_id, paper_title, source_type="PDFPLUMBER_TABLE")
                    raw_rows.extend(rr)
                    clean_rows.extend(cc)

    except Exception as e:
        print("❌ pdfplumber failed:", e)

    # -------- 2) Camelot fallback for ruled tables ----------
    if camelot is not None:
        try:
            tables = camelot.read_pdf(str(pdf), pages="1-5", flavor="stream")
            for tb in tables:
                df = tb.df
                t_text = "\n".join(df.astype(str).apply(lambda r: " ".join(r.values), axis=1).tolist())
                rr, cc = extract_matches(t_text, paper_id, paper_title, source_type="CAMELOT_TABLE")
                raw_rows.extend(rr)
                clean_rows.extend(cc)
        except Exception as e:
            pass

raw_df = pd.DataFrame(raw_rows)
clean_df = pd.DataFrame(clean_rows)

raw_df.to_csv(OUT_RAW, index=False)
clean_df.to_csv(OUT_CLEAN, index=False)

print("\n✅ DONE.")
print("RAW extracted rows:", len(raw_df))
print("CLEAN extracted rows:", len(clean_df))
print("Saved:", OUT_RAW)
print("Saved:", OUT_CLEAN)
