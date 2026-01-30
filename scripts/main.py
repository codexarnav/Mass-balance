import pandas as pd
import numpy as np

# === Load the data ===
df = pd.read_csv('cleaned_model_predictions.csv')

initial_api = df['initial_api_percent']
initial_deg = df['initial_degradant_percent']
model_api_loss = df['model_predicted_api_loss_percent']
model_deg = df['model_predicted_degradant_percent']

api_stressed = 100 - model_api_loss
degradant_stressed = model_deg
initial_total = initial_api + initial_deg

df['smb_mass_balance'] = api_stressed + degradant_stressed

df['amb_mass_balance'] = ((api_stressed + degradant_stressed) / initial_total) * 100


df['amb_deficiency'] = 100 - df['amb_mass_balance']


degradant_increase = degradant_stressed - initial_deg
api_loss = initial_api - api_stressed

df['rmb_mass_balance'] = np.where(
    api_loss != 0,
    (degradant_increase / api_loss) * 100,
    np.nan
)

# 5. Relative Mass Balance Deficiency (RMBD)
df['rmb_deficiency'] = 100 - df['rmb_mass_balance']

# === Save updated CSV ===
df.to_csv("final_model_data_with_mass_balance.csv", index=False)
print("✓ All mass balance calculations complete and saved to final_model_data_with_mass_balance.csv")
