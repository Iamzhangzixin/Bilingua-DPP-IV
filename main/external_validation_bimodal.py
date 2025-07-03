import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
import torch.optim as optim
# import wandb # Not needed for external validation script
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    roc_auc_score, matthews_corrcoef, roc_curve, auc,
    ConfusionMatrixDisplay, confusion_matrix
)
# from sklearn.model_selection import StratifiedKFold # Not needed
import torch.nn.functional as F
# from torch.optim.lr_scheduler import CosineAnnealingLR # Not needed
from sklearn.preprocessing import StandardScaler
import warnings
# import optuna # Not needed
# from optuna import Trial # Not needed
# from optuna.samplers import TPESampler # Not needed
import random
# import joblib # Not needed
# from sklearn.linear_model import LogisticRegression # Not needed
from scipy.special import softmax # Potentially needed if weights were generated differently, but using predefined weights here
# from scipy.optimize import minimize # Not needed
import gc # Import garbage collector
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from pathlib import Path # Use pathlib for cleaner path handling

# --- Dynamic Path Configuration ---
try:
    # Get the directory where the current script is located
    script_dir = Path(__file__).parent.resolve()
except NameError:
     # Fallback for interactive environments
     script_dir = Path.cwd()

# Navigate up one level to get the project root directory (Bilingua-DPPIV)
project_root = script_dir.parent.resolve()

print(f"Detected script directory: {script_dir}")
print(f"Calculated project root: {project_root}")

# --- Configuration for Bimodal External Validation ---
EXTERNAL_CONFIG = {
    # --- Data Paths ---
    "external_csv": project_root / 'data' / 'ExternalValidation.csv',
    "external_seq_feat_path": project_root / 'encode' / 'ESMC' / 'ExternalValidation_esm_6b.npy',
    "external_mol_feat_path": project_root / 'encode' / 'Uni-Mol2' / 'atomic' / 'ExternalValidation_unimol_1.1B_atomic_pad.npy',
    # --- Paths for Scaler Fitting (Original Training Data) ---
    "train_seq_feat_path_for_scaler": project_root / 'encode' / 'ESMC' / 'Train_esm_6b.npy',
    "train_mol_feat_path_for_scaler": project_root / 'encode' / 'Uni-Mol2' / 'atomic' / 'Train_unimol_1.1B_atomic_pad.npy',
    # --- Model & Ensemble ---
    "trained_model_dir": project_root / 'model' / 'Bilingua_2Modal_gated_final_model', # Directory containing saved bimodal fold models
    "num_folds": 10,
    "ensemble_weights": np.array([
        0.10622111, 0.11844834, 0.10952407, 0.08646557, 0.08694772,
        0.10939511, 0.10229512, 0.09935919, 0.09760279, 0.08374098
    ]),
    # --- Prediction Settings ---
    "batch_size": 64,
    "num_workers": 4,
    # --- Plotting Configuration ---
    "plot_dir": project_root / 'plot' / 'ExternalValidation', # Specific subdirectory for these plots
    "plot_dpi": 300,
    "color_primary_pink": (255/255, 96/255, 164/255),
    "color_primary_blue": (30/255, 174/255, 255/255),
    "plot_palette": [(30/255, 174/255, 255/255), (255/255, 96/255, 164/255)], # Blue and Pink
    "sns_palette": sns.color_palette([(30/255, 174/255, 255/255), (255/255, 96/255, 164/255)]),
    # --- Fixed Architecture Params (Should match training - provide defaults) ---
    "final_encoder_fc_dim": 128, # From CNN_bimodal_Final.py CONFIG
    "final_fusion_hidden_dim": 128, # From CNN_bimodal_Final.py CONFIG
    "seq_pooling_size": 5,       # From CNN_bimodal_Final.py CONFIG
    "mol_pooling_size": 15,      # From CNN_bimodal_Final.py CONFIG
}

# Ensure plot subdirectory exists
EXTERNAL_CONFIG["plot_dir"].mkdir(parents=True, exist_ok=True)

# Rich console for better table formatting
console = Console()

# Ignore specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Function to select the best available GPU (Copied) ---
def get_best_gpu():
    """Selects the GPU with the most free memory."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        return torch.device("cpu")
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        print("No CUDA devices found, using CPU.")
        return torch.device("cpu")
    best_device_idx = -1
    max_free_memory = -1
    print(f"Found {num_devices} CUDA device(s). Checking memory...")
    for i in range(num_devices):
        try:
            device_name = torch.cuda.get_device_name(i)
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            free_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            print(f"  Device {i}: {device_name}, Free Memory: {free_gb:.2f} GB / {total_gb:.2f} GB")
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_device_idx = i
        except Exception as e:
            print(f"Could not get memory info for device {i}: {e}")
    if best_device_idx == -1:
        print("Could not determine best GPU, falling back to cuda:0 or CPU if unavailable.")
        return torch.device("cuda:0" if num_devices > 0 else "cpu")
    else:
        print(f"Selected Device {best_device_idx} with {max_free_memory / (1024**3):.2f} GB free memory.")
        return torch.device(f"cuda:{best_device_idx}")

# Set device using the selection function
device = get_best_gpu()
print(f"Using device: {device}")

# --- Model Definitions (Copied *EXACTLY* from CNN_bimodal_Final.py) ---
class EncoderCNN(nn.Module):
    """Generic CNN Encoder with Residual Connections."""
    def __init__(self, embedding_dim, projected_dim, pooling_size,
                 num_classes=2, conv1_out=64, conv2_out=128,
                 kernel_size=3, dropout_rate=0.5): # Match defaults used in training
        super(EncoderCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.projected_dim = projected_dim
        self.pooling_size = pooling_size
        self.pre_cnn_projection = nn.Linear(embedding_dim, projected_dim)
        self.conv1 = nn.Conv1d(in_channels=projected_dim, out_channels=conv1_out,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.res1_conv1 = nn.Conv1d(conv1_out, conv1_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.res1_bn1 = nn.BatchNorm1d(conv1_out)
        self.res1_conv2 = nn.Conv1d(conv1_out, conv1_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.res1_bn2 = nn.BatchNorm1d(conv1_out)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.res2_conv1 = nn.Conv1d(conv2_out, conv2_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.res2_bn1 = nn.BatchNorm1d(conv2_out)
        self.res2_conv2 = nn.Conv1d(conv2_out, conv2_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.res2_bn2 = nn.BatchNorm1d(conv2_out)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.pooling_size)
        fc1_input_dim = conv2_out * self.pooling_size
        encoder_fc_dim = EXTERNAL_CONFIG["final_encoder_fc_dim"] # Use config value
        self.fc1 = nn.Linear(fc1_input_dim, encoder_fc_dim)
        # Note: Dropout is part of the encoder *output* in the MultiModal model, not within EncoderCNN itself

    def forward(self, x):
        x_proj = self.pre_cnn_projection(x)
        x_permuted = x_proj.permute(0, 2, 1)
        out1 = F.relu(self.bn1(self.conv1(x_permuted)))
        identity1 = out1
        res1 = F.relu(self.res1_bn1(self.res1_conv1(out1)))
        res1 = self.res1_bn2(self.res1_conv2(res1))
        if res1.shape != identity1.shape:
             diff = identity1.shape[2] - res1.shape[2]
             res1 = F.pad(res1, (diff // 2, diff - diff // 2))
        out1 = F.relu(identity1 + res1)
        out1_pooled = self.pool(out1)
        out2 = F.relu(self.bn2(self.conv2(out1_pooled)))
        identity2 = out2
        res2 = F.relu(self.res2_bn1(self.res2_conv1(out2)))
        res2 = self.res2_bn2(self.res2_conv2(res2))
        if res2.shape != identity2.shape:
             diff = identity2.shape[2] - res2.shape[2]
             res2 = F.pad(res2, (diff // 2, diff - diff // 2))
        out2 = F.relu(identity2 + res2)
        out2_pooled = self.pool(out2)
        out_pooled = self.adaptive_pool(out2_pooled)
        out_flat = out_pooled.view(out_pooled.size(0), -1)
        out_repr = F.relu(self.fc1(out_flat))
        return out_repr

class MultiModalFusionModel(nn.Module):
    # IMPORTANT: Ensure this definition MATCHES EXACTLY the one used during training
    def __init__(self, seq_embedding_dim, mol_embedding_dim, # Original dims
                 seq_projected_dim, mol_projected_dim, # Projected CNN dims
                 fusion_method='concat', # Default if not loaded
                 num_classes=2, conv1_out=64, conv2_out=128, kernel_size=3,
                 dropout_rate=0.5, n_heads=4): # Default n_heads
        super(MultiModalFusionModel, self).__init__()

        self.fusion_method = fusion_method
        self.n_heads = n_heads
        encoder_output_dim = EXTERNAL_CONFIG["final_encoder_fc_dim"]
        fusion_hidden_dim = EXTERNAL_CONFIG["final_fusion_hidden_dim"]

        # === Encoders ===
        self.seq_encoder = EncoderCNN(
            embedding_dim=seq_embedding_dim, projected_dim=seq_projected_dim,
            pooling_size=EXTERNAL_CONFIG["seq_pooling_size"],
            num_classes=num_classes, conv1_out=conv1_out, conv2_out=conv2_out,
            kernel_size=kernel_size, dropout_rate=dropout_rate # Pass dropout here if needed by EncoderCNN
        )
        self.mol_encoder = EncoderCNN(
            embedding_dim=mol_embedding_dim, projected_dim=mol_projected_dim,
            pooling_size=EXTERNAL_CONFIG["mol_pooling_size"],
            num_classes=num_classes, conv1_out=conv1_out, conv2_out=conv2_out,
            kernel_size=kernel_size, dropout_rate=dropout_rate # Pass dropout here if needed by EncoderCNN
        )

        # === Fusion Layers ===
        if fusion_method == 'concat':
            self.fusion_fc = nn.Linear(encoder_output_dim * 2, fusion_hidden_dim)
        elif fusion_method == 'attention':
            self.attn = nn.MultiheadAttention(
                embed_dim=encoder_output_dim,
                num_heads=n_heads,
                batch_first=True,
                dropout=dropout_rate # Dropout within attention
            )
            self.fusion_fc = nn.Linear(encoder_output_dim * 2, fusion_hidden_dim)
        elif fusion_method == 'gated':
            self.gate_seq = nn.Linear(encoder_output_dim, encoder_output_dim)
            self.gate_mol = nn.Linear(encoder_output_dim, encoder_output_dim)
            self.fusion_fc = nn.Linear(encoder_output_dim, fusion_hidden_dim)
        else:
            # Fallback or error if fusion method from checkpoint is unexpected
            print(f"Warning: Unsupported fusion method '{fusion_method}' found in checkpoint. Defaulting to 'concat'.")
            self.fusion_method = 'concat'
            self.fusion_fc = nn.Linear(encoder_output_dim * 2, fusion_hidden_dim)

        # Final layers
        self.fusion_dropout = nn.Dropout(dropout_rate) # Dropout after fusion FC
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)

    def forward(self, seq_features, mol_features):
        batch_size = seq_features.size(0)
        seq_repr = self.seq_encoder(seq_features)
        mol_repr = self.mol_encoder(mol_features)

        # === Apply Fusion ===
        fused = None # Initialize
        if self.fusion_method == 'concat':
            fused_repr = torch.cat([seq_repr, mol_repr], dim=1)
            fused = F.relu(self.fusion_fc(fused_repr))
        elif self.fusion_method == 'attention':
            combined = torch.stack([seq_repr, mol_repr], dim=1)
            attn_out, _ = self.attn(combined, combined, combined)
            attn_out_flat = attn_out.reshape(batch_size, -1)
            fused = F.relu(self.fusion_fc(attn_out_flat))
        elif self.fusion_method == 'gated':
            gate_seq = torch.sigmoid(self.gate_seq(seq_repr))
            gate_mol = torch.sigmoid(self.gate_mol(mol_repr))
            gated_sum = (gate_seq * seq_repr) + (gate_mol * mol_repr)
            fused = F.relu(self.fusion_fc(gated_sum))
        else: # Should not happen if handled in init, but as safeguard
             fused_repr = torch.cat([seq_repr, mol_repr], dim=1)
             fused = F.relu(self.fusion_fc(fused_repr)) # Default to concat logic


        fused_dropped = self.fusion_dropout(fused)
        output = self.classifier(fused_dropped)
        return output

# --- Performance Metrics Function (Copied) ---
def performance(y_true, y_predict, y_prob=None):
    """Calculates various performance metrics."""
    TP = np.sum((y_true == 1) & (y_predict == 1))
    TN = np.sum((y_true == 0) & (y_predict == 0))
    FP = np.sum((y_true == 0) & (y_predict == 1))
    FN = np.sum((y_true == 1) & (y_predict == 0))

    acc = accuracy_score(y_true, y_predict) * 100
    sn = TP / (TP + FN) if (TP + FN) != 0 else 0.0 # Sensitivity (Recall)
    sp = TN / (FP + TN) if (FP + TN) != 0 else 0.0 # Specificity
    pre = TP / (TP + FP) if (TP + FP) != 0 else 0.0 # Precision

    mcc = matthews_corrcoef(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)

    auc_score_val = 0.0
    if y_prob is not None:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
             y_prob_positive = y_prob[:, 1]
        elif y_prob.ndim == 1:
             y_prob_positive = y_prob
        else:
             print(f"Warning: Unexpected y_prob shape {y_prob.shape} for AUC calculation.")
             y_prob_positive = None

        if y_prob_positive is not None:
            try:
                auc_score_val = roc_auc_score(y_true, y_prob_positive)
            except ValueError as e:
                print(f"Warning: ROC AUC calculation failed: {e}. Setting AUC to 0.")
                auc_score_val = 0.0
    return acc, sn, sp, pre, mcc, f1, auc_score_val

# --- Data Loading Function for Bimodal External Validation ---
def load_bimodal_validation_data(csv_path, seq_feat_path, mol_feat_path):
    """Loads external bimodal features, labels, and the full dataframe."""
    print(f"Loading external data from: {csv_path}")
    print(f"Loading external sequence features from: {seq_feat_path}")
    print(f"Loading external molecule features from: {mol_feat_path}")
    try:
        external_data_df = pd.read_csv(csv_path)
        external_labels = external_data_df['label'].values
        external_seq_features = np.load(seq_feat_path)
        external_mol_features = np.load(mol_feat_path)
    except FileNotFoundError as e:
        print(f"Error loading external data file: {e}.")
        raise
    except Exception as e:
        print(f"An error occurred during external data loading: {e}")
        raise

    assert external_seq_features.shape[0] == external_mol_features.shape[0] == len(external_labels), "External feature/label count mismatch"
    assert 'seq' in external_data_df.columns, "'seq' column not found in external CSV file."

    print(f"External sequence features shape: {external_seq_features.shape}")
    print(f"External molecule features shape: {external_mol_features.shape}")
    print("External data loading complete.")
    return external_data_df, external_seq_features, external_mol_features, external_labels

# --- Visualization Functions (Adapted for Bimodal External Validation) ---
def plot_roc_curve_external(y_true, y_prob, filename, title):
    """Plots the ROC curve for the ensemble prediction on the external set."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1]) # Prob of positive class
        roc_auc = auc(fpr, tpr)

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color=EXTERNAL_CONFIG["color_primary_pink"], lw=2.5,
                 label=f'Ensemble ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Chance', alpha=0.8)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        # Ensure parent directory exists before saving
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=EXTERNAL_CONFIG["plot_dpi"])
        plt.close()
        print(f"Saved ROC curve plot to {filename}")
    except Exception as e:
        print(f"Could not generate ROC curve plot '{title}': {e}")

def plot_confusion_matrix_external(y_true, y_pred, display_labels, filename, title):
    """Plots and saves the confusion matrix."""
    if y_true is None or y_pred is None:
        print(f"Skipping confusion matrix plot '{title}': Missing true labels or predictions.")
        return
    try:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

        fig, ax = plt.subplots(figsize=(6, 6))
        cmap_blue = sns.light_palette(EXTERNAL_CONFIG["color_primary_blue"], as_cmap=True)
        disp.plot(cmap=cmap_blue, ax=ax, values_format='d')
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=EXTERNAL_CONFIG["plot_dpi"])
        plt.close(fig)
        print(f"Saved confusion matrix plot to {filename}")
    except Exception as e:
        print(f"Could not generate confusion matrix plot '{title}': {e}")

def plot_probability_distribution_external(y_true, y_prob_positive, filename, title):
    """Plots and saves the probability distribution for the positive class."""
    if y_true is None or y_prob_positive is None:
        print(f"Skipping probability distribution plot '{title}': Missing true labels or probabilities.")
        return
    try:
        df = pd.DataFrame({
            'Probability (Class 1)': y_prob_positive,
            'True Label': [f"Class {label}" for label in y_true]
        })
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Probability (Class 1)', hue='True Label',
                     kde=True, stat='density', common_norm=False, palette=EXTERNAL_CONFIG["sns_palette"])
        plt.title(title, fontsize=14)
        plt.xlabel('Predicted Probability for Class 1', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.tight_layout()
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=EXTERNAL_CONFIG["plot_dpi"])
        plt.close()
        print(f"Saved probability distribution plot to {filename}")
    except Exception as e:
        print(f"Could not generate probability distribution plot '{title}': {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n===== Starting Bimodal External Validation =====")

    # --- 1. Load External Validation Data ---
    try:
        external_data_df, external_seq_features, external_mol_features, external_labels = load_bimodal_validation_data(
            EXTERNAL_CONFIG["external_csv"],
            EXTERNAL_CONFIG["external_seq_feat_path"],
            EXTERNAL_CONFIG["external_mol_feat_path"]
        )
        external_labels_tensor = torch.LongTensor(external_labels)
    except Exception as e:
        print(f"Fatal Error: Failed to load external data. Aborting. Error: {e}")
        exit()

    # --- 2. Load Original Training Data for Scaler Fitting ---
    try:
        print(f"Loading original training features for scaler fitting...")
        train_seq_features_orig = np.load(EXTERNAL_CONFIG['train_seq_feat_path_for_scaler'])
        train_mol_features_orig = np.load(EXTERNAL_CONFIG['train_mol_feat_path_for_scaler'])
        print(f"Original training sequence features shape: {train_seq_features_orig.shape}")
        print(f"Original training molecule features shape: {train_mol_features_orig.shape}")
    except Exception as e:
        print(f"Fatal Error: Failed to load original training features for scalers. Aborting. Error: {e}")
        exit()

    # --- 3. Fit Scalers and Scale External Data ---
    print("Fitting Scalers on original training data...")
    # Sequence Scaler
    seq_scaler = StandardScaler()
    train_seq_shape = train_seq_features_orig.shape
    seq_scaler.fit(train_seq_features_orig.reshape(-1, train_seq_shape[-1]))
    # Molecule Scaler
    mol_scaler = StandardScaler()
    train_mol_shape = train_mol_features_orig.shape
    mol_scaler.fit(train_mol_features_orig.reshape(-1, train_mol_shape[-1]))

    print("Scalers fitted. Transforming external data...")
    # Scale Sequence Features
    external_seq_shape = external_seq_features.shape
    external_seq_features_scaled = seq_scaler.transform(
        external_seq_features.reshape(-1, external_seq_shape[-1])
    ).reshape(external_seq_shape)
    # Scale Molecule Features
    external_mol_shape = external_mol_features.shape
    external_mol_features_scaled = mol_scaler.transform(
        external_mol_features.reshape(-1, external_mol_shape[-1])
    ).reshape(external_mol_shape)

    print("External data scaling complete.")
    external_seq_tensor = torch.Tensor(external_seq_features_scaled)
    external_mol_tensor = torch.Tensor(external_mol_features_scaled)

    # Clean up original training features from memory
    del train_seq_features_orig, train_mol_features_orig
    gc.collect()

    # --- 4. Load Models and Predict ---
    all_fold_external_probs = []
    print(f"\nLoading {EXTERNAL_CONFIG['num_folds']} fold models from: {EXTERNAL_CONFIG['trained_model_dir']}")

    # Get feature dimensions from loaded data
    seq_embedding_dim_actual = external_seq_features_scaled.shape[-1]
    mol_embedding_dim_actual = external_mol_features_scaled.shape[-1]

    for fold in range(1, EXTERNAL_CONFIG['num_folds'] + 1):
        print(f"--- Processing Fold {fold}/{EXTERNAL_CONFIG['num_folds']} ---")
        model_path = EXTERNAL_CONFIG['trained_model_dir'] / f'fold_{fold}.model'
        model_load = None # Ensure model is None if loading fails

        if not model_path.exists():
            print(f"Warning: Model file not found for fold {fold} at {model_path}. Skipping fold.")
            all_fold_external_probs.append(None)
            continue

        try:
            checkpoint = torch.load(model_path, map_location=device)
            params = checkpoint.get('params', {}) # Get saved hyperparameters

            # Extract necessary parameters, using defaults from EXTERNAL_CONFIG if not found
            seq_projected_dim_loaded = params.get('seq_proj_dim', 128) # Example default
            mol_projected_dim_loaded = params.get('mol_proj_dim', 64)  # Example default
            conv1_out_loaded = params.get('conv1_out', 64)
            conv2_out_loaded = params.get('conv2_out', 128)
            kernel_size_loaded = params.get('kernel_size', 3)
            dropout_rate_loaded = params.get('dropout_rate', 0.5)
            fusion_method_loaded = params.get('fusion_method', 'gated') # Default to gated as per dir name
            n_heads_loaded = params.get('n_heads', 4) # Default for attention

            # Re-instantiate model using loaded parameters
            model_load = MultiModalFusionModel(
                seq_embedding_dim=seq_embedding_dim_actual, # Use actual dimension
                mol_embedding_dim=mol_embedding_dim_actual, # Use actual dimension
                seq_projected_dim=seq_projected_dim_loaded,
                mol_projected_dim=mol_projected_dim_loaded,
                fusion_method=fusion_method_loaded,
                num_classes=2,
                conv1_out=conv1_out_loaded, conv2_out=conv2_out_loaded,
                kernel_size=kernel_size_loaded, dropout_rate=dropout_rate_loaded,
                n_heads=n_heads_loaded
            ).to(device)

            model_load.load_state_dict(checkpoint['model_state_dict'])
            model_load.eval()
            print(f"  Model for fold {fold} loaded successfully (Fusion: {fusion_method_loaded}).")

            # Prepare external dataset loader
            external_dataset = TensorDataset(external_seq_tensor, external_mol_tensor, external_labels_tensor)
            external_loader = DataLoader(external_dataset, batch_size=EXTERNAL_CONFIG["batch_size"],
                                         shuffle=False, pin_memory=True, num_workers=EXTERNAL_CONFIG["num_workers"])

            # Predict on external data
            fold_probs_batches = []
            with torch.no_grad():
                for seq_feat_ext, mol_feat_ext, _ in external_loader:
                    seq_feat_ext, mol_feat_ext = seq_feat_ext.to(device), mol_feat_ext.to(device)
                    outputs_ext = model_load(seq_feat_ext, mol_feat_ext) # Pass both features
                    probs_ext = F.softmax(outputs_ext, dim=1)
                    fold_probs_batches.append(probs_ext.cpu().numpy())

            fold_external_probs = np.concatenate(fold_probs_batches)
            all_fold_external_probs.append(fold_external_probs)
            print(f"  Predictions for fold {fold} completed.")

        except Exception as e:
            print(f"Error processing fold {fold}: {e}")
            all_fold_external_probs.append(None)
        finally:
            del model_load, checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- 5. Ensemble Predictions ---
    successful_fold_indices = [i for i, p in enumerate(all_fold_external_probs) if p is not None]
    num_successful_folds = len(successful_fold_indices)
    print(f"\nNumber of successfully loaded folds for prediction: {num_successful_folds}/{EXTERNAL_CONFIG['num_folds']}")

    if num_successful_folds == 0:
        print("Fatal Error: No models could be loaded or predictions failed for all folds. Aborting.")
        exit()

    valid_fold_probs = [p for p in all_fold_external_probs if p is not None]
    valid_probs_np = np.array(valid_fold_probs)

    # Use the predefined weights
    original_weights_all_folds = EXTERNAL_CONFIG['ensemble_weights']
    ensemble_probs = None
    ensemble_preds = None
    ensemble_method = 'weighted_avg' # Explicitly using this method

    if len(original_weights_all_folds) != EXTERNAL_CONFIG['num_folds']:
        print(f"Error: Number of provided weights ({len(original_weights_all_folds)}) does not match expected folds ({EXTERNAL_CONFIG['num_folds']}). Aborting.")
        exit() # Stop if weights are incorrect

    if num_successful_folds < EXTERNAL_CONFIG['num_folds']:
        print(f"Warning: Only {num_successful_folds}/{EXTERNAL_CONFIG['num_folds']} folds were loaded successfully.")
        print("Using weights corresponding to successful folds and renormalizing...")
        selected_weights = original_weights_all_folds[successful_fold_indices]
        normalized_selected_weights = selected_weights / np.sum(selected_weights)
        print(f"  Selected weights: {selected_weights}")
        print(f"  Normalized weights: {normalized_selected_weights}")
        ensemble_method = 'weighted_avg (renormalized)'
        ensemble_probs = np.average(valid_probs_np, axis=0, weights=normalized_selected_weights)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
    else:
        print("Applying 'weighted_avg' ensemble method using pre-calculated weights...")
        print(f"  Using weights: {original_weights_all_folds}")
        ensemble_probs = np.average(valid_probs_np, axis=0, weights=original_weights_all_folds)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)


    # --- 6. Calculate and Display Metrics and Detailed Results ---
    print(f"\n===== Bimodal External Validation Performance ({ensemble_method.upper()} Ensemble) =====")
    try:
        ext_acc, ext_sn, ext_sp, ext_pre, ext_mcc, ext_f1, ext_auc = performance(
            external_labels, ensemble_preds, ensemble_probs
        )

        print(f"  Accuracy: {ext_acc:.2f}%")
        print(f"  Sensitivity (Recall): {ext_sn:.4f}")
        print(f"  Specificity: {ext_sp:.4f}")
        print(f"  Precision: {ext_pre:.4f}")
        print(f"  MCC: {ext_mcc:.4f}")
        print(f"  F1-Score: {ext_f1:.4f}")
        print(f"  AUC: {ext_auc:.4f}")
        print("-" * 71)

        print("\nDetailed Prediction Results:")
        results_df = pd.DataFrame({
            'Sequence': external_data_df['seq'],
            'True Label': external_labels,
            'Predicted Label': ensemble_preds,
            'Predicted Probability (Class 1)': ensemble_probs[:, 1].round(4)
        })
        print(results_df.to_string())

        print("=======================================================================")

    except Exception as e:
        print(f"Error calculating performance metrics or generating results table: {e}")

    # --- 7. Generate Visualizations ---
    print("\nGenerating visualizations...")
    plot_output_dir = EXTERNAL_CONFIG['plot_dir']
    plot_suffix = f"_bimodal_ExternalValidation_{ensemble_method.replace(' ', '_').replace('(','').replace(')','')}.png"

    # Plot ROC Curve
    plot_roc_curve_external(
        external_labels, ensemble_probs,
        filename=plot_output_dir / f"roc_curve{plot_suffix}",
        title=f"ROC Curve (Bimodal External Validation - {ensemble_method.upper()} Ensemble)"
    )

    # Plot Confusion Matrix
    plot_confusion_matrix_external(
        external_labels, ensemble_preds,
        display_labels=["Non-inhibitor", "Inhibitor"],
        filename=plot_output_dir / f"confusion_matrix{plot_suffix}",
        title=f"Confusion Matrix (Bimodal External Validation - {ensemble_method.upper()} Ensemble)"
    )

    # Plot Probability Distribution
    plot_probability_distribution_external(
        external_labels, ensemble_probs[:, 1],
        filename=plot_output_dir / f"probability_dist{plot_suffix}",
        title=f"Probability Distribution (Bimodal External Validation - {ensemble_method.upper()} Ensemble)"
    )

    print("\nScript finished.")
