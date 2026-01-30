import pandas as pd
from scholarly import scholarly
import time

print("✅ Starting Scholar fetch...")

OUT_PATH = "stress_library/02_literature/scholar_results.csv"

queries = [
    'stability indicating method "forced degradation" HCl NaOH H2O2',
    '"forced degradation study" 0.1 N HCl 60°C HPLC',
    '"oxidative degradation" 3% H2O2 "stability indicating"',
    '"photolytic degradation" "1.2 million lux" ICH Q1B',
    '"forced degradation" thermal 80°C 24 h HPLC'
]

MAX_PER_QUERY = 6

results = []
paper_id = 1

for q in queries:
    print(f"\n🔍 Query: {q}")
    search = scholarly.search_pubs(q)

    count = 0
    for pub in search:
        try:
            bib = pub.get("bib", {})

            title = bib.get("title", "")
            year = bib.get("pub_year", "")
            authors = bib.get("author", "")
            venue = bib.get("venue", "")
            citations = pub.get("num_citations", 0)

            pub_url = pub.get("pub_url", "")
            eprint_url = pub.get("eprint_url", "")

            results.append({
                "paper_id": f"P{paper_id:03d}",
                "title": title,
                "year": year,
                "authors": authors,
                "venue": venue,
                "citation_count": citations,
                "pub_url": pub_url,
                "eprint_url": eprint_url
            })

            print("   ✅", title[:70])

            paper_id += 1
            count += 1

            time.sleep(4)  # Avoid Google blocking

            if count >= MAX_PER_QUERY:
                break

        except Exception as e:
            print("⚠️ Skipping due to error:", e)
            continue

df = pd.DataFrame(results)

df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

df.to_csv(OUT_PATH, index=False)
print(f"\n✅ Saved {len(df)} papers into: {OUT_PATH}")
print("✅ Done.")
