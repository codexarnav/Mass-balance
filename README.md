# Mass Balance & Force Degradation Prediction System

This project implements a comprehensive system for predicting API loss and degradant formation under various stress conditions, evaluating mass balance formulas, and applying heuristic analysis to determine the most appropriate mass balance method.

## System Overview

The system combines Graph Neural Networks (GNN) for molecular structure analysis and Transformer-based models (BERT) for processing stress conditions to predict degradation outcomes. It further evaluates these predictions using rigorous mass balance formulas and heuristics.

### Workflow Pipeline

1.  **Data Preparation**: Integration of drug molecular data with stress condition data.
2.  **Model Training**: Training a hybrid Deep Learning model to predict API loss and degradant percentage.
3.  **Inference**: generating predictions on test datasets.
4.  **Mass Balance Evaluation**: Calculating Standard (SMB), Absolute (AMB), and Relative (RMB) Mass Balance metrics.
5.  **Heuristic Analysis**: Categorizing cases based on deficiency thresholds to recommend the optimal mass balance formula.

---

## Project Structure

- **`scripts/`**
    - `model_train.py`: Main training script for the hybrid GNN-BERT model.
    - `test.py`: Inference script for generating predictions and evaluating model performance.
    - `main.py`: Post-processing script for calculating mass balance metrics (SMB, AMB, RMB).
    - `heuristics.py`: Analysis script for applying heuristic rules to label data points.

---

## Detailed Component Description

### 1. Model Training (`model_train.py`)
The core prediction engine uses a multi-modal architecture:
- **GNNModel**: Uses `GATv2Conv` layers to process molecular graphs (SMILES) and extract structural embeddings.
- **StressEncoder**: Combines `BERT` (for text descriptions of stress/reagents) and an `ANN` (for numeric conditions like temperature, pH).
- **FusionModule**: Merges the molecular and stress embeddings to predict:
    - Predicted API Loss (%)
    - Predicted Degradant (%)

**Key Features:**
- Handles missing values and NaNs robustly.
- Saves checkpoints and the best model (`best_model.pth`).

### 2. Inference & Testing (`test.py`)
Performs inference using the trained model:
- Loads `best_model.pth`.
- Generates predictions for the validation/test set.
- Comparison with ground truth values.
- Outputs `model_predictions_complete.csv` containing both actual and predicted values.

### 3. Mass Balance Evaluation (`main.py`)
Evaluates the physical consistency of predictions using Mass Balance equations:
- **SMB (Standard Mass Balance)**: `API_stressed + Degradant_stressed`
- **AMB (Absolute Mass Balance)**: Percentage of total mass recovered relative to initial mass.
- **RMB (Relative Mass Balance)**: Ratio of degradant increase to API loss.
- Calculates **Deficiencies** (AMBD, RMBD) to quantify deviations.
- Outputs `final_model_data_with_mass_balance.csv`.

### 4. Heuristic Analysis (`heuristics.py`)
Determines the regime of degradation and appropriate formula usage:
- Inputs: Mass balance deficiencies and stress conditions.
- **Logic**:
    - **AMB**: Low API loss (<5%).
    - **AMBD**: High deficiency cases (AMB deficiency >= 15%, High API loss, Specific stress conditions like strong base or photolytic loss).
    - **RMBD**: Cases where RMB deficiency is within acceptable thresholds (e.g., < 70th percentile).
- Outputs `labeled_dataset_final.csv` with the final `heuristic_label`.

---

## Requirements

- Python 3.8+
- PyTorch & PyTorch Geometric
- Transformers (Hugging Face)
- RDKit
- Scikit-learn
- Pandas, NumPy

## Usage

1.  **Train the Model**:
    ```bash
    python scripts/model_train.py
    ```
2.  **Run Inference**:
    ```bash
    python scripts/test.py
    ```
3.  **Calculate Mass Balance**:
    ```bash
    python scripts/main.py
    ```
4.  **Run Heuristics**:
    ```bash
    python scripts/heuristics.py
    ```
