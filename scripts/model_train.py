import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.nn import Module, Linear, Dropout, ReLU, LayerNorm
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem
from torch_geometric.data import Data as GeoData, Batch
from transformers import BertModel, BertTokenizer
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("LOADING & CLEANING DATA")
print("="*80)

# Load both datasets
drug_data = pd.read_csv('model_data.csv')
stress_data = pd.read_csv('ann_data.csv')

print(f"\nOriginal drug data shape: {drug_data.shape}")
print(f"Original stress data shape: {stress_data.shape}")

# Combine to ensure row alignment
combined_df = pd.concat([drug_data, stress_data], axis=1)

# Define essential columns
essential_columns = [
    'smiles', 'stress_type', 'reagent', 'severity',
    'temp_bin', 'duration_bin', 'strength_bin',
    'temperature_mean', 'duration_mean', 'strength_mean',
    'pH_mean', 'predicted_api_loss_percent'
]

# Drop rows with any missing values in key fields
before_drop = combined_df.shape[0]
combined_df.dropna(subset=essential_columns, inplace=True)
after_drop = combined_df.shape[0]

print(f"Dropped {before_drop - after_drop} rows due to missing values")

# Split back into drug and stress DataFrames
# Fix to ensure all required columns are preserved
drug_data = combined_df.loc[:, combined_df.columns.isin(['smiles'])].reset_index(drop=True)
stress_data = combined_df.drop(columns=['smiles']).reset_index(drop=True)


stress_data = stress_data.loc[:, ~stress_data.columns.duplicated()]


print(f"After cleaning: drug_data = {drug_data.shape}, stress_data = {stress_data.shape}")

# Print some info about NaNs if any still linger
print("\nRemaining NaNs:")
print("Drug SMILES NaN:", drug_data['smiles'].isna().sum())
print("Numeric NaN:", stress_data[['temperature_mean', 'duration_mean', 'strength_mean', 'pH_mean']].isna().sum().sum())
print("Target NaN:", stress_data['predicted_api_loss_percent'].isna().sum())




numeric_features = stress_data[['temperature_mean', 'duration_mean', 'strength_mean', 'pH_mean']]
scaler = StandardScaler()
scaler.fit(numeric_features)

class GNNModel(Module):
    def __init__(self, node_feat_dim=4, embedding_dim=256):
        super().__init__()
        self.conv1 = GATv2Conv(node_feat_dim, 64, heads=4, concat=True, dropout=0.2)
        self.conv2 = GATv2Conv(256, 64, heads=2, concat=True, dropout=0.2)
        self.conv3 = GATv2Conv(128, embedding_dim, heads=1, concat=False)
        self.enhancer = torch.nn.Sequential(
            Linear(embedding_dim, embedding_dim * 2),
            ReLU(),
            Dropout(0.3),
            Linear(embedding_dim * 2, embedding_dim),
            LayerNorm(embedding_dim)
        )

    def forward(self, x, edge_index, batch):
        # Check for NaN in input
        if torch.isnan(x).any():
            print("WARNING: NaN detected in GNN input!")
            x = torch.nan_to_num(x, 0.0)
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = self.enhancer(global_mean_pool(x, batch))
        
        # Check output
        if torch.isnan(x).any():
            print("WARNING: NaN detected in GNN output!")
        
        return x


class ANNModel(Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = Linear(hidden_dim // 2, 128)
        self.norm = LayerNorm(128)

    def forward(self, x):
        # Check for NaN in input
        if torch.isnan(x).any():
            print("WARNING: NaN detected in ANN input!")
            x = torch.nan_to_num(x, 0.0)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.norm(x)
        
        # Check output
        if torch.isnan(x).any():
            print("WARNING: NaN detected in ANN output!")
        
        return x

class StressEncoder(Module):
    def __init__(self, text_model, ann_model, tokenizer, scaler):
        super().__init__()
        self.text_model = text_model
        self.ann_model = ann_model
        self.tokenizer = tokenizer
        self.scaler = scaler
        
        # Freeze BERT
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(self, text_inputs, numeric_inputs):
        with torch.no_grad():
            encoding = self.tokenizer(text_inputs, return_tensors='pt', padding=True,
                                      truncation=True, max_length=32).to(device)
            bert_output = self.text_model(**encoding).last_hidden_state[:, 0, :]
        
        # Scale numeric inputs
        numeric_scaled = self.scaler.transform(numeric_inputs)
        numeric_scaled = torch.tensor(numeric_scaled, dtype=torch.float32).to(device)
        
        # Check for NaN
        if torch.isnan(bert_output).any():
            print("WARNING: NaN in BERT output!")
        if torch.isnan(numeric_scaled).any():
            print("WARNING: NaN in scaled numeric features!")
            numeric_scaled = torch.nan_to_num(numeric_scaled, 0.0)
        
        combined = torch.cat([numeric_scaled, bert_output], dim=1)
        return self.ann_model(combined)

class FusionModule(Module):
    def __init__(self, gnn_dim, stress_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = Linear(gnn_dim + stress_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = Linear(hidden_dim // 2, 64)
        self.api_head = Linear(64, 1)
        self.degradant_head = Linear(64, 1)


    def forward(self, gnn_emb, stress_emb):
        x = torch.cat([gnn_emb, stress_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc3(x))
        api_output = self.api_head(x).squeeze()
        degr_output = self.degradant_head(x).squeeze()
        return api_output, degr_output

def smiles_to_graph(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        
        def atom_feats(a):
            return torch.tensor([
                float(a.GetAtomicNum()), 
                float(a.GetTotalDegree()), 
                float(a.GetFormalCharge()), 
                float(a.GetIsAromatic())
            ], dtype=torch.float32)
        
        x = torch.stack([atom_feats(a) for a in mol.GetAtoms()])
        
        # Check for NaN in atom features
        if torch.isnan(x).any():
            print(f"WARNING: NaN in atom features for SMILES: {smile}")
            return None
        
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([(i, j), (j, i)])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
        
        return GeoData(x=x, edge_index=edge_index)
    except Exception as e:
        print(f"Error converting SMILES {smile}: {e}")
        return None

class DrugStressDataset(Dataset):
    def __init__(self, drug_df, stress_df):
        self.drug_df = drug_df.reset_index(drop=True)
        self.stress_df = stress_df.reset_index(drop=True)
        
        assert len(self.drug_df) == len(self.stress_df), "Drug and stress data must have same length!"
        
        print(f"Dataset created with {len(self.drug_df)} samples")
    
    def __len__(self):
        return len(self.drug_df)
    
    def __getitem__(self, idx):
        drug_row = self.drug_df.iloc[idx]
        smile = drug_row['smiles']
        graph = smiles_to_graph(smile)
        
        # Get stress data
        stress_row = self.stress_df.iloc[idx]
        text_input = ' '.join([
            str(stress_row['stress_type']),
            str(stress_row['reagent']),
            str(stress_row['severity']),
            str(stress_row['temp_bin']),
            str(stress_row['duration_bin']),
            str(stress_row['strength_bin'])
        ])
        
        numeric_input = np.array([
            float(stress_row['temperature_mean']),
            float(stress_row['duration_mean']),
            float(stress_row['strength_mean']),
            float(stress_row['pH_mean'])
        ], dtype=np.float32)
        
        
        if np.isnan(numeric_input).any():
            print(f"WARNING: NaN in numeric input at index {idx}")
            numeric_input = np.nan_to_num(numeric_input, 0.0)
        
        target_api = float(stress_row['predicted_api_loss_percent'])
        target_degradant = float(stress_row['predicted_degradant_percent'])
        return graph, text_input, numeric_input, target_api, target_degradant


def collate_fn(batch):
    graphs, texts, numerics, targets_api, targets_deg = zip(*batch)
    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    if len(valid_indices) == 0:
        return None
    graphs = [graphs[i] for i in valid_indices]
    texts = [texts[i] for i in valid_indices]
    numerics = np.stack([numerics[i] for i in valid_indices])
    targets_api = torch.tensor([targets_api[i] for i in valid_indices], dtype=torch.float32)
    targets_deg = torch.tensor([targets_deg[i] for i in valid_indices], dtype=torch.float32)
    graph_batch = Batch.from_data_list(graphs)
    return graph_batch, list(texts), numerics, targets_api, targets_deg

print("\n" + "="*80)
print("SPLITTING DATA")
print("="*80)

indices = np.arange(len(drug_data))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_drug_data = drug_data.iloc[train_idx].reset_index(drop=True)
train_stress_data = stress_data.iloc[train_idx].reset_index(drop=True)

val_drug_data = drug_data.iloc[val_idx].reset_index(drop=True)
val_stress_data = stress_data.iloc[val_idx].reset_index(drop=True)

print(f"\nTraining samples: {len(train_drug_data)}")
print(f"Validation samples: {len(val_drug_data)}")

train_dataset = DrugStressDataset(train_drug_data, train_stress_data)
val_dataset = DrugStressDataset(val_drug_data, val_stress_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)  # Reduced batch size
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# === Initialize models ===

print("\n" + "="*80)
print("INITIALIZING MODELS")
print("="*80)

print("\nLoading BERT model...")
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Initializing GNN model...")
gnn_model = GNNModel().to(device)

print("Initializing ANN model...")
ann_input_dim = 4 + 768
ann_model = ANNModel(ann_input_dim).to(device)

print("Initializing Stress Encoder...")
stress_encoder = StressEncoder(bert_model, ann_model, tokenizer, scaler).to(device)

print("Initializing Fusion Module...")
fusion_model = FusionModule(gnn_dim=256, stress_dim=128).to(device)

print(f"\nAll models initialized on {device}")

total_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
total_params += sum(p.numel() for p in ann_model.parameters() if p.requires_grad)
total_params += sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

optimizer = torch.optim.Adam(
    list(gnn_model.parameters()) + 
    list(ann_model.parameters()) + 
    list(fusion_model.parameters()),
    lr=5e-4,  # Reduced learning rate
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

criterion = torch.nn.MSELoss()

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

EPOCHS = 40
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    gnn_model.train()
    ann_model.train()
    fusion_model.train()

    train_loss = 0.0
    train_batches = 0

    for batch_idx, batch_data in enumerate(train_loader):
        if batch_data is None:
            continue

        batch_graphs, batch_texts, batch_numerics, target_api, target_deg = batch_data
        batch_graphs = batch_graphs.to(device)
        target_api = target_api.to(device)
        target_deg = target_deg.to(device)

        if torch.isnan(target_api).any() or torch.isnan(target_deg).any():
            print(f"WARNING: NaN in batch targets at batch {batch_idx}")
            continue

        optimizer.zero_grad()

        try:
            gnn_emb = gnn_model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
            stress_emb = stress_encoder(batch_texts, batch_numerics)
            pred_api, pred_deg = fusion_model(gnn_emb, stress_emb)

            if torch.isnan(pred_api).any() or torch.isnan(pred_deg).any():
                print(f"WARNING: NaN in predictions at batch {batch_idx}, skipping...")
                continue

            loss_api = criterion(pred_api, target_api)
            loss_deg = criterion(pred_deg, target_deg)
            loss = loss_api + loss_deg

            if torch.isnan(loss):
                print(f"WARNING: NaN loss at batch {batch_idx}, skipping...")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(ann_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - API Loss: {loss_api.item():.4f}, Degradant Loss: {loss_deg.item():.4f}")

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    avg_train_loss = train_loss / train_batches if train_batches > 0 else float('nan')

    gnn_model.eval()
    ann_model.eval()
    fusion_model.eval()

    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            if batch_data is None:
                continue

            batch_graphs, batch_texts, batch_numerics, target_api, target_deg = batch_data
            batch_graphs = batch_graphs.to(device)
            target_api = target_api.to(device)
            target_deg = target_deg.to(device)

            if torch.isnan(target_api).any() or torch.isnan(target_deg).any():
                continue

            try:
                gnn_emb = gnn_model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
                stress_emb = stress_encoder(batch_texts, batch_numerics)
                pred_api, pred_deg = fusion_model(gnn_emb, stress_emb)

                if torch.isnan(pred_api).any() or torch.isnan(pred_deg).any():
                    continue

                loss_api = criterion(pred_api, target_api)
                loss_deg = criterion(pred_deg, target_deg)
                loss = loss_api + loss_deg

                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batches += 1
            except Exception as e:
                continue
    
    avg_val_loss = val_loss / val_batches if val_batches > 0 else float('nan')

    if not np.isnan(avg_val_loss):
        scheduler.step(avg_val_loss)

    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"{'='*80}\n")

    if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch + 1,
            'gnn_state': gnn_model.state_dict(),
            'ann_state': ann_model.state_dict(),
            'fusion_state': fusion_model.state_dict(),
            'scaler': scaler,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, 'best_model.pth')
        print(f"✓ Best model saved! (Val Loss: {avg_val_loss:.4f})\n")

    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch + 1,
            'gnn_state': gnn_model.state_dict(),
            'ann_state': ann_model.state_dict(),
            'fusion_state': fusion_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler': scaler,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, f'checkpoint_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pth\n")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"\nBest Validation Loss: {best_val_loss:.4f}")
print(f"Best model saved as: best_model.pth")
