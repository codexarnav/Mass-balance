import pandas as pd



df = pd.read_csv("main_dataset.csv")

required_cols = [
    "true_api_loss_percent",
    "rmb_deficiency",
    "amb_deficiency",
    "stress_type",
    "severity"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")


print("\nRMB deficiency distribution:")
print(df["rmb_deficiency"].describe())

print("\nAMB deficiency distribution:")
print(df["amb_deficiency"].describe())

print("\nAPI loss distribution:")
print(df["true_api_loss_percent"].describe())




rmb_q70 = df["rmb_deficiency"].quantile(0.70)

print(f"\nRMBD threshold (70th percentile of RMB deficiency): {rmb_q70:.2f}")


def heuristic_label(row):
    api_loss = row["true_api_loss_percent"]
    rmb_def = row["rmb_deficiency"]
    amb_def = row["amb_deficiency"]
    stress = str(row["stress_type"]).lower()
    severity = str(row["severity"]).lower()


    if api_loss < 5:
        return "AMB"

    if amb_def >= 15:
        return "AMBD"

    if api_loss >= 25:
        return "AMBD"

    # Strong base fragmentation
    if "base" in stress and severity == "strong":
        return "AMBD"

    # Photolytic physical loss
    if "photo" in stress and amb_def >= 10:
        return "AMBD"

    if rmb_def < rmb_q70:
        return "RMBD"

    # Remaining severe collapse
    return "AMBD"

df["heuristic_label"] = df.apply(heuristic_label, axis=1)
print("\nFinal label distribution:")
print(df["heuristic_label"].value_counts())


output_path = "labeled_dataset_final.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved labeled dataset to: {output_path}")
