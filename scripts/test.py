import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
from transformers import BertTokenizer, BertModel
from model_train import GNNModel, ANNModel, StressEncoder, FusionModule, DrugStressDataset, collate_fn
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("LOADING & PREPROCESSING DATA")
print("="*80)

# === Load both datasets ===
drug_data = pd.read_csv('model_data.csv')
stress_data = pd.read_csv('ann_data.csv')

print(f"\nOriginal drug data shape: {drug_data.shape}")
print(f"Original stress data shape: {stress_data.shape}")

# === Combine to ensure row alignment ===
combined_df = pd.concat([drug_data, stress_data], axis=1)
print(f"Combined shape: {combined_df.shape}")

# === Check if stress_type exists ===
if 'stress_type' not in combined_df.columns:
    print("\n⚠️  WARNING: 'stress_type' column not found!")
    print("Available columns:", list(combined_df.columns))
    print("\nAttempting to infer stress_type from other columns...")
    
    # Create a dummy stress_type column if it doesn't exist
    if 'reagent' in combined_df.columns:
        combined_df['stress_type'] = 'chemical'  # Default stress type
        print("✓ Created default 'stress_type' column")

# === Define essential columns ===
essential_columns = [
    'smiles', 'stress_type', 'reagent', 'severity',
    'temp_bin', 'duration_bin', 'strength_bin',
    'temperature_mean', 'duration_mean', 'strength_mean',
    'pH_mean', 'predicted_api_loss_percent', 'predicted_degradant_percent'
]

# Check which columns are missing
missing_cols = [col for col in essential_columns if col not in combined_df.columns]
if missing_cols:
    print(f"\n⚠️  ERROR: Missing required columns: {missing_cols}")
    print("Available columns:", list(combined_df.columns))
    raise ValueError(f"Missing required columns: {missing_cols}")

# === Drop rows with missing values ===
before_drop = combined_df.shape[0]
combined_df.dropna(subset=essential_columns, inplace=True)
after_drop = combined_df.shape[0]
print(f"\nDropped {before_drop - after_drop} rows due to missing values")
print(f"Remaining samples: {after_drop}")

# === Reset index BEFORE splitting ===
combined_df = combined_df.reset_index(drop=True)

# === IMPORTANT: Store the complete combined_df for later reference ===
complete_data_backup = combined_df.copy()

# === Split back into drug and stress DataFrames (EXACTLY like training) ===
drug_data = combined_df.loc[:, combined_df.columns.isin(['smiles'])].reset_index(drop=True)
stress_data = combined_df.drop(columns=['smiles']).reset_index(drop=True)

# === CRITICAL: Remove duplicate columns ===
stress_data = stress_data.loc[:, ~stress_data.columns.duplicated()]

print(f"\nAfter splitting:")
print(f"  Drug data shape: {drug_data.shape}")
print(f"  Stress data shape: {stress_data.shape}")

# === Verify no NaN in critical columns ===
print("\nVerifying data integrity:")
print(f"  Drug SMILES NaN: {drug_data['smiles'].isna().sum()}")
print(f"  Stress numeric NaN: {stress_data[['temperature_mean', 'duration_mean', 'strength_mean', 'pH_mean']].isna().sum().sum()}")
print(f"  Target API NaN: {stress_data['predicted_api_loss_percent'].isna().sum()}")
print(f"  Target Degradant NaN: {stress_data['predicted_degradant_percent'].isna().sum()}")

# === Validation split (SAME as training with same random_state) ===
from sklearn.model_selection import train_test_split
_, val_idx = train_test_split(np.arange(len(drug_data)), test_size=1000, random_state=42)

val_drug_data = drug_data.iloc[val_idx].reset_index(drop=True)
val_stress_data = stress_data.iloc[val_idx].reset_index(drop=True)
val_complete_data = complete_data_backup.iloc[val_idx].reset_index(drop=True)

print(f"\nValidation samples: {len(val_drug_data)}")

# === Load scaler from checkpoint ===
print("\n" + "="*80)
print("LOADING MODEL CHECKPOINT")
print("="*80)

try:
    checkpoint = torch.load('best_model.pth', map_location=device)
    scaler = checkpoint['scaler']
    print("✓ Checkpoint loaded successfully")
    print(f"  Trained on epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Training loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    raise

# === Initialize models ===
print("\n" + "="*80)
print("INITIALIZING MODELS")
print("="*80)

try:
    print("Loading BERT model...")
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("Initializing GNN model...")
    gnn_model = GNNModel().to(device)
    
    print("Initializing ANN model...")
    ann_model = ANNModel(input_dim=4 + 768).to(device)
    
    print("Initializing Stress Encoder...")
    stress_encoder = StressEncoder(bert_model, ann_model, tokenizer, scaler).to(device)
    
    print("Initializing Fusion Module...")
    fusion_model = FusionModule(gnn_dim=256, stress_dim=128).to(device)
    
    print(f"✓ All models initialized on {device}")
except Exception as e:
    print(f"❌ Error initializing models: {e}")
    raise

# === Load weights ===
print("\nLoading model weights...")
try:
    gnn_model.load_state_dict(checkpoint['gnn_state'])
    ann_model.load_state_dict(checkpoint['ann_state'])
    fusion_model.load_state_dict(checkpoint['fusion_state'])
    print("✓ Model weights loaded successfully")
except Exception as e:
    print(f"❌ Error loading model weights: {e}")
    raise

# === Set to evaluation mode ===
gnn_model.eval()
ann_model.eval()
fusion_model.eval()

# === Create DataLoader ===
print("\n" + "="*80)
print("CREATING VALIDATION DATASET")
print("="*80)

try:
    val_dataset = DrugStressDataset(val_drug_data, val_stress_data)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"✓ Validation dataset created: {len(val_dataset)} samples")
    print(f"✓ Validation batches: {len(val_loader)}")
except Exception as e:
    print(f"❌ Error creating dataset: {e}")
    raise

# === Inference ===
print("\n" + "="*80)
print("RUNNING INFERENCE")
print("="*80 + "\n")

# Track which samples have been processed
predictions_list = []
processed_indices = []
batch_count = 0
total_samples = 0
errors = 0
current_idx = 0  # Track the current sample index

with torch.no_grad():
    for batch_idx, batch_data in enumerate(val_loader):
        if batch_data is None:
            errors += 1
            print(f"  Batch {batch_idx+1}: Skipped (None batch)")
            # Still need to increment current_idx for skipped batches
            current_idx += 32  # Assuming batch size of 32
            continue

        try:
            batch_graphs, batch_texts, batch_numerics, true_api, true_deg = batch_data
            batch_graphs = batch_graphs.to(device)
            true_api = true_api.cpu().numpy()
            true_deg = true_deg.cpu().numpy()

            # Forward pass
            gnn_emb = gnn_model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
            stress_emb = stress_encoder(batch_texts, batch_numerics)
            pred_api, pred_deg = fusion_model(gnn_emb, stress_emb)

            # Handle scalar vs 1D tensor
            if pred_api.dim() == 0:
                pred_api = pred_api.unsqueeze(0)
            if pred_deg.dim() == 0:
                pred_deg = pred_deg.unsqueeze(0)

            pred_api = pred_api.cpu().numpy()
            pred_deg = pred_deg.cpu().numpy()

            # Store results with corresponding indices
            batch_size = len(pred_api)
            for i in range(batch_size):
                sample_idx = current_idx + i
                processed_indices.append(sample_idx)
                
                predictions_list.append({
                    "model_predicted_api_loss_percent": round(float(pred_api[i]), 2),
                    "model_predicted_degradant_percent": round(float(pred_deg[i]), 2),
                    "true_api_loss_percent": round(float(true_api[i]), 2),
                    "true_degradant_percent": round(float(true_deg[i]), 2),
                    "initial_api_percent": 100.0,
                    "initial_degradant_percent": 0.0
                })
            
            current_idx += batch_size
            batch_count += 1
            total_samples += batch_size
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx+1}/{len(val_loader)} batches, {total_samples} samples...")
                
        except Exception as e:
            errors += 1
            print(f"  Batch {batch_idx+1}: Error - {str(e)}")
            # Still need to increment current_idx for failed batches
            current_idx += 32  # Assuming batch size of 32
            continue

print(f"\n{'='*80}")
print("INFERENCE COMPLETE")
print(f"{'='*80}")
print(f"Total batches processed: {batch_count}")
print(f"Total samples: {total_samples}")
print(f"Errors encountered: {errors}")

# === Combine predictions with original data ===
if len(predictions_list) > 0:
    print("\n" + "="*80)
    print("COMBINING PREDICTIONS WITH ORIGINAL DATA")
    print("="*80)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions_list)
    
    # Get the corresponding complete data for processed samples
    processed_complete_data = val_complete_data.iloc[processed_indices].reset_index(drop=True)
    
    # Combine everything
    output_df = pd.concat([processed_complete_data, predictions_df], axis=1)
    
    # Reorder columns for better readability - put key info first
    key_columns = []
    
    # Drug identification columns
    if 'drugbank_id' in output_df.columns:
        key_columns.append('drugbank_id')
    if 'name' in output_df.columns:
        key_columns.append('name')
    key_columns.append('smiles')
    
    # Stress condition columns
    stress_condition_cols = ['stress_type', 'reagent', 'severity', 
                            'temp_bin', 'duration_bin', 'strength_bin',
                            'temperature_mean', 'duration_mean', 'strength_mean', 'pH_mean']
    for col in stress_condition_cols:
        if col in output_df.columns:
            key_columns.append(col)
    
    # Initial values
    key_columns.extend(['initial_api_percent', 'initial_degradant_percent'])
    
    # True values (ground truth)
    key_columns.extend(['true_api_loss_percent', 'true_degradant_percent'])
    
    # Model predictions
    key_columns.extend(['model_predicted_api_loss_percent', 'model_predicted_degradant_percent'])
    
    # Get remaining columns
    remaining_cols = [col for col in output_df.columns if col not in key_columns]
    
    # Reorder
    final_column_order = key_columns + remaining_cols
    output_df = output_df[final_column_order]
    
    # Save to CSV
    output_df.to_csv("model_predictions_complete.csv", index=False)
    
    print(f"\n✓ Complete predictions saved to model_predictions_complete.csv")
    print(f"✓ Total columns: {len(output_df.columns)}")
    print(f"✓ Total samples: {len(output_df)}")
    
    print(f"\nColumn summary:")
    print(f"  Key columns: {len(key_columns)}")
    print(f"  Additional columns: {len(remaining_cols)}")
    
    print(f"\nFirst few rows of complete output:")
    print(output_df.head())
    
    print(f"\nPrediction statistics:")
    stats_cols = ['model_predicted_api_loss_percent', 'model_predicted_degradant_percent', 
                  'true_api_loss_percent', 'true_degradant_percent']
    print(output_df[stats_cols].describe())
    
    # Calculate prediction errors
    output_df['api_prediction_error'] = abs(output_df['model_predicted_api_loss_percent'] - output_df['true_api_loss_percent'])
    output_df['degradant_prediction_error'] = abs(output_df['model_predicted_degradant_percent'] - output_df['true_degradant_percent'])
    
    print(f"\nPrediction error statistics:")
    print(output_df[['api_prediction_error', 'degradant_prediction_error']].describe())
    
    # Save again with error columns
    output_df.to_csv("model_predictions_complete.csv", index=False)
    print(f"\n✓ Updated CSV with prediction errors")
    
else:
    print("\n❌ No predictions generated - check for errors above")