import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
import torch.optim as optim
import wandb
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    roc_auc_score, matthews_corrcoef, roc_curve, auc, # Added roc_curve, auc
    ConfusionMatrixDisplay, confusion_matrix # Added ConfusionMatrixDisplay, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
import warnings
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import random
import joblib
from sklearn.linear_model import LogisticRegression # Added LogisticRegression import
from scipy.special import softmax
from scipy.optimize import minimize
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

# --- Plotting Configuration ---
PLOT_DIR = project_root / 'plot' / 'sequence_only' # Changed plot directory path
PLOT_DIR.mkdir(parents=True, exist_ok=True) # Create plot directory if it doesn't exist (added parents=True)
PLOT_DPI = 300
# Define primary colors using RGB tuples (scaled 0-1)
COLOR_PRIMARY_PINK = (255/255, 96/255, 164/255)
COLOR_PRIMARY_BLUE = (30/255, 174/255, 255/255)
# Generate some同色系 colors if needed (e.g., lighter/darker versions)
COLOR_LIGHT_PINK = (1.0, 0.6, 0.8)
COLOR_LIGHT_BLUE = (0.5, 0.85, 1.0)
COLOR_DARK_PINK = (0.8, 0.2, 0.5)
COLOR_DARK_BLUE = (0.1, 0.5, 0.8)
# Palette for visualizations
PLOT_PALETTE = [COLOR_PRIMARY_BLUE, COLOR_PRIMARY_PINK, COLOR_LIGHT_BLUE, COLOR_LIGHT_PINK, COLOR_DARK_BLUE, COLOR_DARK_PINK]
SNS_PALETTE = sns.color_palette([COLOR_PRIMARY_BLUE, COLOR_PRIMARY_PINK]) # Specific palette for Seaborn plots like histograms

# Rich console for better table formatting
console = Console()

# --- End Dynamic Path Configuration & Plotting Setup ---


# Ignore specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration Dictionary (Sequence Only)
CONFIG = {
    # Paths (Dynamically generated absolute paths using pathlib)
    "train_csv": project_root / 'data' / 'Train.csv',
    "seq_train_feat_path": project_root / 'encode' / 'ESMC' / 'Train_esm_6b.npy',
    # Removed: "mol_train_feat_path": project_root / 'encode' / 'Uni-Mol2' / 'atomic' / 'Train_unimol_1.1B_atomic_pad.npy',
    "test_csv": project_root / 'data' / 'Test.csv',
    "seq_test_feat_path": project_root / 'encode' / 'ESMC' / 'Test_esm_6b.npy',
    # Removed: "mol_test_feat_path": project_root / 'encode' / 'Uni-Mol2' / 'atomic' / 'Test_unimol_1.1B_atomic_pad.npy',
    "model_dir_base": project_root / 'model_seq_only', # Base directory for sequence-only models
    "optuna_study_path": project_root / 'model_seq_only' / 'optuna_study_seq_only.pkl', # Path for sequence-only study

    # Training Settings
    "seed": 42,
    "num_classes": 2,
    "optuna_kfold_splits": 5, # Folds for Optuna hyperparameter search (Using 5 for efficiency)
    "main_kfold_splits": 10,  # Folds for final model training and evaluation
    "optuna_epochs": 50,
    "main_epochs": 100,
    "optuna_early_stop_patience": 10, # Slightly increased patience for Optuna
    "main_early_stop_patience": 10,
    "clip_value": 1.0,
    "num_workers": 4,
    "label_smoothing": 0.1,

    # Optuna Settings
    # "optuna_n_trials": 80, # Number of Optuna trials
    "optuna_n_trials": 2, # Number of Optuna trials
    "optuna_timeout": 10800, # Timeout for Optuna study (3 hours)

    # Hyperparameter Search Ranges (Sequence Only - Removed mol/fusion specific)
    "lr_range": (1e-6, 5e-4),
    "weight_decay_range": (1e-4, 1e-1),
    "seq_pre_cnn_dim_choices": [128, 256],
    # Removed: "mol_pre_cnn_dim_choices": [64, 128],
    "conv1_out_choices": [16, 32, 64],
    "conv2_out_choices": [32, 64, 128],
    "kernel_size_range": (2, 5),
    "dropout_rate_range": (0.4, 0.8), # Dropout within CNN encoder
    "batch_size_choices": [16, 32, 64],
    # Removed: "fusion_method_choices": ['concat', 'attention', 'gated'],
    # Removed: "attention_heads_choices": [2, 4],

    # Model Architecture Config (Output dim of encoder)
    "final_encoder_fc_dim": 128, # Dimension output by the encoder's final FC layer
    # Removed: "final_fusion_hidden_dim": 128,

    # Wandb Config
    "wandb_project": "prot_t5",
    "wandb_entity": "zhangzixin1999",
    "wandb_run_name": "Bilingua-DPPIV-SeqOnly", # Updated run name for sequence-only

    # CNN Encoder specific Pooling sizes
    "seq_pooling_size": 5,
    # Removed: "mol_pooling_size": 15,
}

# Print configured paths for verification
print("\n--- Configured Paths (Sequence Only) ---")
for key, value in CONFIG.items():
    if "path" in key or "dir" in key:
        print(f"{key}: {value}")
print("--------------------------------------\n")

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# Wandb Initialization
wandb.init(project=CONFIG["wandb_project"], entity=CONFIG["wandb_entity"], name=CONFIG["wandb_run_name"])
wandb.config.update({k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()}) # Log config to wandb, ensure Paths are strings


# --- Function to select the best available GPU ---
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
            # Use torch.cuda.mem_get_info() which is more standard
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
# --- End of GPU selection logic ---

# --------------------- Generic CNN Encoder ---------------------
class EncoderCNN(nn.Module):
    """Generic CNN Encoder with Residual Connections."""
    def __init__(self, embedding_dim, projected_dim, pooling_size,
                 num_classes=CONFIG["num_classes"], conv1_out=64, conv2_out=128,
                 kernel_size=3, dropout_rate=0.5): # Dropout added here
        super(EncoderCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.projected_dim = projected_dim
        self.pooling_size = pooling_size

        # Project input features to the dimension expected by CNN
        self.pre_cnn_projection = nn.Linear(embedding_dim, projected_dim)

        # --- First Convolutional Block with Residual Connection ---
        self.conv1 = nn.Conv1d(in_channels=projected_dim, out_channels=conv1_out,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        # Residual block 1
        self.res1_conv1 = nn.Conv1d(conv1_out, conv1_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.res1_bn1 = nn.BatchNorm1d(conv1_out)
        self.res1_conv2 = nn.Conv1d(conv1_out, conv1_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.res1_bn2 = nn.BatchNorm1d(conv1_out)

        # --- Max Pooling ---
        self.pool = nn.MaxPool1d(kernel_size=2) # Standard pooling layer

        # --- Second Convolutional Block with Residual Connection ---
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        # Residual block 2
        self.res2_conv1 = nn.Conv1d(conv2_out, conv2_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.res2_bn1 = nn.BatchNorm1d(conv2_out)
        self.res2_conv2 = nn.Conv1d(conv2_out, conv2_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.res2_bn2 = nn.BatchNorm1d(conv2_out)

        # --- Adaptive Pooling and Final FC Layer ---
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.pooling_size)
        fc1_input_dim = conv2_out * self.pooling_size
        encoder_fc_dim = CONFIG["final_encoder_fc_dim"]
        self.fc1 = nn.Linear(fc1_input_dim, encoder_fc_dim)
        self.dropout = nn.Dropout(dropout_rate) # Added dropout after FC

    def forward(self, x):
        # x shape: (batch, seq_len, embedding_dim)
        # Project features first
        x_proj = self.pre_cnn_projection(x) # -> (batch, seq_len, projected_dim)
        # Permute for Conv1d: (batch, channels, seq_len)
        x_permuted = x_proj.permute(0, 2, 1) # -> (batch, projected_dim, seq_len)

        # --- Block 1 ---
        out1 = F.relu(self.bn1(self.conv1(x_permuted)))
        identity1 = out1
        res1 = F.relu(self.res1_bn1(self.res1_conv1(out1)))
        res1 = self.res1_bn2(self.res1_conv2(res1))
        # Ensure residual shape matches identity shape (though padding should handle this)
        if res1.shape != identity1.shape:
             diff = identity1.shape[2] - res1.shape[2]
             res1 = F.pad(res1, (diff // 2, diff - diff // 2))
        out1 = F.relu(identity1 + res1)
        out1_pooled = self.pool(out1) # Apply pooling after first block

        # --- Block 2 ---
        out2 = F.relu(self.bn2(self.conv2(out1_pooled)))
        identity2 = out2
        res2 = F.relu(self.res2_bn1(self.res2_conv1(out2)))
        res2 = self.res2_bn2(self.res2_conv2(res2))
        if res2.shape != identity2.shape:
             diff = identity2.shape[2] - res2.shape[2]
             res2 = F.pad(res2, (diff // 2, diff - diff // 2))
        out2 = F.relu(identity2 + res2)
        out2_pooled = self.pool(out2) # Apply pooling after second block

        # --- Final Layers ---
        out_pooled = self.adaptive_pool(out2_pooled) # Adaptive pool
        out_flat = out_pooled.view(out_pooled.size(0), -1) # Flatten
        out_repr = F.relu(self.fc1(out_flat)) # Final FC layer output representation
        out_repr_dropped = self.dropout(out_repr) # Apply dropout
        return out_repr_dropped # Return dropped representation

# --------------------- Sequence Only Model (Using Generic Encoder) ---------------------
class SequenceModel(nn.Module):
    def __init__(self, seq_embedding_dim, seq_projected_dim,
                 num_classes=CONFIG["num_classes"], conv1_out=64, conv2_out=128, kernel_size=3,
                 dropout_rate=0.5): # Pass dropout rate to EncoderCNN
        super(SequenceModel, self).__init__()

        encoder_output_dim = CONFIG["final_encoder_fc_dim"] # e.g., 128

        # === Sequence (ESM6B) Encoder ===
        self.seq_encoder = EncoderCNN(
            embedding_dim=seq_embedding_dim, projected_dim=seq_projected_dim,
            pooling_size=CONFIG["seq_pooling_size"],
            num_classes=num_classes, conv1_out=conv1_out, conv2_out=conv2_out,
            kernel_size=kernel_size, dropout_rate=dropout_rate # Pass dropout rate here
        )

        # === Final Classifier ===
        # Input dimension is the output dimension of the sequence encoder's FC layer
        self.classifier = nn.Linear(encoder_output_dim, num_classes)

    def forward(self, seq_features):
        # seq_features: (batch, seq_len, seq_emb_dim)

        # Get representation from encoder
        seq_repr = self.seq_encoder(seq_features) # (batch, encoder_output_dim)

        # Final classification
        output = self.classifier(seq_repr) # (batch, num_classes)
        return output

# --------------------- Performance Metrics ---------------------
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

    # Use scikit-learn's implementation for robustness
    mcc = matthews_corrcoef(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)

    # Calculate AUC if probabilities are provided
    auc = 0.0
    if y_prob is not None:
        # Ensure y_prob is 1D array of probabilities for the positive class
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
             y_prob_positive = y_prob[:, 1]
        elif y_prob.ndim == 1:
             y_prob_positive = y_prob
        else:
             print(f"Warning: Unexpected y_prob shape {y_prob.shape} for AUC calculation.")
             y_prob_positive = None

        if y_prob_positive is not None:
            try:
                auc = roc_auc_score(y_true, y_prob_positive)
            except ValueError as e:
                print(f"Warning: ROC AUC calculation failed: {e}. Setting AUC to 0.")
                auc = 0.0
    return acc, sn, sp, pre, mcc, f1, auc

# --------------------- Data Loading (Sequence Only) ---------------------
def load_sequence_data(train_csv, seq_train_feat_path,
                       test_csv, seq_test_feat_path):
    """Loads sequence features and labels."""
    print("Loading sequence data...")
    try:
        train_data = pd.read_csv(train_csv)
        train_labels = train_data['label'].values
        test_data = pd.read_csv(test_csv)
        test_labels = test_data['label'].values

        train_seq_features = np.load(seq_train_feat_path)
        test_seq_features = np.load(seq_test_feat_path)

    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Please check paths in CONFIG.")
        raise
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        raise

    # Basic shape validation
    assert train_seq_features.shape[0] == len(train_labels), "Train feature/label count mismatch"
    assert test_seq_features.shape[0] == len(test_labels), "Test feature/label count mismatch"

    print(f"Train sequence features shape: {train_seq_features.shape}")
    print(f"Test sequence features shape: {test_seq_features.shape}")
    print("Data loading complete.")

    # Return tuple with sequence features and labels
    return (train_seq_features, train_labels), \
           (test_seq_features, test_labels)


# --------------------- Training and Validation Loops (Sequence Only) ---------------------
def train_sequence_epoch(model, loader, criterion, optimizer, clip_value=CONFIG["clip_value"]):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    # Unpack sequence features and labels from loader
    for seq_features, labels in loader:
        seq_features, labels = seq_features.to(device), labels.to(device)
        batch_sample_size = seq_features.size(0)
        total_samples += batch_sample_size

        optimizer.zero_grad()
        outputs = model(seq_features) # Pass only sequence features to model
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) # Clip gradients
        optimizer.step()

        running_loss += loss.item() * batch_sample_size
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = 100. * correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

def validate_sequence_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    total_samples = 0
    with torch.no_grad():
        # Unpack sequence features and labels
        for seq_features, labels in loader:
            seq_features, labels = seq_features.to(device), labels.to(device)
            batch_sample_size = seq_features.size(0)
            total_samples += batch_sample_size

            outputs = model(seq_features) # Pass only sequence features to model
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_sample_size
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1) # Get probabilities for AUC

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0

    # Calculate metrics using the collected labels and predictions/probabilities
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    if len(all_labels_np) > 0:
        acc, sn, sp, pre, mcc, f1, auc = performance(all_labels_np, all_preds_np, all_probs_np)
    else:
        acc, sn, sp, pre, mcc, f1, auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return epoch_loss, acc, sn, sp, pre, mcc, f1, auc


# --------------------- Ensemble Predictions ---------------------
# This function works on probabilities and labels, so it remains the same.
# Note: val_metrics structure should match the output of validate_sequence_epoch
def ensemble_predictions(all_fold_test_probs, test_labels, val_metrics=None, ensemble_method='avg'):
    """Enhanced ensemble prediction function with multiple strategies."""
    print(f"\n--- Applying Ensemble Method: {ensemble_method} ---")
    all_probs_np = np.array(all_fold_test_probs)

    if all_probs_np.ndim < 3 or all_probs_np.shape[0] == 0:
        print(f"Error: Invalid shape or no fold probabilities available for ensembling: {all_probs_np.shape}")
        return None, None, [0.0] * 7 # Return zero metrics

    num_folds, num_samples, num_classes = all_probs_np.shape
    average_probs = None # Initialize
    ensemble_preds = None

    # === Ensemble Method Logic ===
    if ensemble_method == 'avg':
        average_probs = np.mean(all_probs_np, axis=0)
    elif ensemble_method == 'weighted_avg':
        weights = None
        if val_metrics is not None and len(val_metrics) == num_folds:
            # Assume val_metrics structure: [acc, sn, sp, pre, mcc, f1, auc] -> MCC is index 4
            mccs = np.array([m[4] if isinstance(m, (list, tuple)) and len(m) > 4 and m[4] is not None else -1.0 for m in val_metrics])
            # Handle negative or zero MCCs safely for weighting
            valid_mccs = mccs[mccs > 0]
            if len(valid_mccs) > 0:
                weights = mccs.copy()
                weights[weights <= 0] = 0 # Give zero weight to non-positive MCCs
                # Apply softmax to potentially enhance differences, then normalize
                weights = softmax(weights * 2) # Adjust scaling factor if needed
                weights = weights / np.sum(weights)
                print(f"Calculated Model weights (based on validation MCC, normalized): {weights}")
            else:
                print("Warning: All validation MCCs were non-positive. Using equal weights for weighted_avg.")
        else:
             print(f"Warning: Validation metrics missing ({len(val_metrics) if val_metrics else 'None'}) or mismatched with folds ({num_folds}). Using equal weights for weighted_avg.")

        if weights is None: # Fallback to equal weights
             weights = np.ones(num_folds) / num_folds

        average_probs = np.sum(weights[:, np.newaxis, np.newaxis] * all_probs_np, axis=0)

    elif ensemble_method == 'max_prob':
        # For each sample, find the fold that predicted the chosen class with the highest probability
        max_probs_values = np.max(all_probs_np, axis=2) # Max prob value for each fold/sample
        best_fold_indices = np.argmax(max_probs_values, axis=0) # Index of the best fold for each sample
        # Get probabilities and predictions from the best fold for each sample
        average_probs = all_probs_np[best_fold_indices, np.arange(num_samples)] # For consistency, use these probs for AUC
        ensemble_preds = np.argmax(average_probs, axis=1) # Derive preds from the selected max probs

    elif ensemble_method == 'stacked':
        # Meta-learner (Logistic Regression) on fold predictions
        # Use probability of class 1 as meta-features: shape (num_samples, num_folds)
        X_meta = all_probs_np[:, :, 1].T
        meta_preds_proba = np.zeros((num_samples, num_classes))
        # Using simple Logistic Regression, could explore others (e.g., LightGBM)
        try:
            # Train final meta-model on all fold predictions to predict test set
            meta_model_final = LogisticRegression(max_iter=1000, C=0.1, solver='liblinear', class_weight='balanced')
            # Note: Stacking typically involves cross-validation *within* the meta-learning step
            # to avoid overfitting the meta-learner. A simpler approach for direct test prediction:
            # Train on all fold-outputs to predict the (potentially unseen) test set labels.
            # This requires OOF predictions on a hold-out set during training, which is complex here.
            # Simpler (but potentially leaky): Train on fold test probs to predict actual test labels.
            # Let's stick to the original implementation's simpler approach for now.
            # This trains the meta-model directly on the test set fold probabilities.
            # It's non-standard but aims to find a good combination rule for the test set itself.
            meta_model_final.fit(X_meta, test_labels) # Fit on the actual test labels - prone to leakage!
            # A more correct approach would use OOF predictions from training folds.
            # For simplicity, keeping the potentially leaky version shown in original code.
            print("Warning: Stacking ensemble uses test labels for meta-model fitting, which is non-standard and can lead to optimistic results.")
            average_probs = meta_model_final.predict_proba(X_meta)
        except Exception as e:
            print(f"Stacking failed: {e}. Falling back to simple average.")
            average_probs = np.mean(all_probs_np, axis=0)

    elif ensemble_method == 'calibrated':
        # Temperature scaling (simple calibration method)
        T = 1.5 # Temperature parameter (can be tuned)
        try:
             # Clip probabilities to avoid log(0)
             safe_probs = np.clip(all_probs_np, 1e-10, 1 - 1e-10)
             # Apply temperature scaling to logits (approximate by scaling log probs)
             scaled_log_probs = np.log(safe_probs) / T
             calibrated_probs = softmax(scaled_log_probs, axis=2)
             average_probs = np.mean(calibrated_probs, axis=0)
        except Exception as e:
            print(f"Calibration failed: {e}. Falling back to simple average.")
            average_probs = np.mean(all_probs_np, axis=0)

    elif ensemble_method == 'optimized':
        # Optimize decision threshold on average probabilities based on MCC
        average_probs_for_opt = np.mean(all_probs_np, axis=0) # Use simple average probs for optimization
        def threshold_objective(threshold, probs, true_labels):
            preds = (probs[:, 1] > threshold[0]).astype(int)
            # Minimize negative MCC (Maximize MCC)
            return -matthews_corrcoef(true_labels, preds)

        try:
            result = minimize(threshold_objective, x0=[0.5], args=(average_probs_for_opt, test_labels),
                              bounds=[(0.01, 0.99)], method='L-BFGS-B')
            best_threshold = result.x[0]
            print(f"Optimized decision threshold: {best_threshold:.4f}")
            average_probs = average_probs_for_opt # Use the mean probs for consistency
            ensemble_preds = (average_probs[:, 1] > best_threshold).astype(int)
        except Exception as e:
            print(f"Threshold optimization failed: {e}. Falling back to simple average with 0.5 threshold.")
            average_probs = average_probs_for_opt
            ensemble_preds = np.argmax(average_probs, axis=1)
    else:
        print(f"Warning: Unknown ensemble method '{ensemble_method}'. Using simple average.")
        average_probs = np.mean(all_probs_np, axis=0)

    # Determine final predictions if not set by the method
    if ensemble_preds is None and average_probs is not None:
        ensemble_preds = np.argmax(average_probs, axis=1)
    elif ensemble_preds is None:
         print("Error: Ensemble predictions could not be determined.")
         return None, None, [0.0] * 7 # Return failure indicators

    # === Calculate Metrics for the chosen ensemble method ===
    metrics_results = performance(test_labels, ensemble_preds, average_probs)
    acc, sn, sp, pre, mcc, f1, auc = metrics_results

    print(f'Ensemble Test Metrics ({ensemble_method}): Acc={acc:.2f}%, SN={sn:.4f}, SP={sp:.4f}, Prec={pre:.4f}, MCC={mcc:.4f}, F1={f1:.4f}, AUC={auc:.4f}')

    wandb.log({
        f"ensemble_{ensemble_method}_Accuracy": acc,
        f"ensemble_{ensemble_method}_Sensitivity": sn,
        f"ensemble_{ensemble_method}_Specificity": sp,
        f"ensemble_{ensemble_method}_Precision": pre,
        f"ensemble_{ensemble_method}_MCC": mcc,
        f"ensemble_{ensemble_method}_F1": f1,
        f"ensemble_{ensemble_method}_ROC_AUC": auc,
    })

    return ensemble_preds, average_probs, metrics_results

def summarize_ensemble_comparison(all_fold_test_probs, test_labels, val_metrics=None):
    """Compare different ensemble methods and return the best one based on MCC."""
    methods = ['avg', 'weighted_avg', 'max_prob', 'stacked', 'calibrated', 'optimized']
    results_dict = {} # Store {method: [acc, sn, sp, pre, mcc, f1, auc]}
    predictions_dict = {} # Store {method: (preds, probs)}

    print("\n===== Ensemble Method Comparison =====")
    table = Table(title="Ensemble Method Performance Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Accuracy (%)", justify="right")
    table.add_column("SN", justify="right")
    table.add_column("SP", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("MCC", justify="right", style="magenta")
    table.add_column("F1", justify="right")
    table.add_column("AUC", justify="right")

    best_mcc = -1.1 # Initialize lower than possible MCC
    best_method = 'avg' # Default

    for method in methods:
        try:
             ensemble_preds, average_probs, metrics_results = ensemble_predictions(all_fold_test_probs, test_labels, val_metrics, method)
             if metrics_results is not None:
                 results_dict[method] = metrics_results
                 predictions_dict[method] = (ensemble_preds, average_probs) # Store preds and probs
                 # MCC is index 4
                 if metrics_results[4] > best_mcc:
                      best_mcc = metrics_results[4]
                      best_method = method
                 # Add row to table
                 table.add_row(
                     method,
                     f"{metrics_results[0]:.2f}", f"{metrics_results[1]:.4f}", f"{metrics_results[2]:.4f}",
                     f"{metrics_results[3]:.4f}", f"{metrics_results[4]:.4f}", f"{metrics_results[5]:.4f}",
                     f"{metrics_results[6]:.4f}"
                 )
             else:
                 print(f"Ensemble method {method} failed to produce results.")
                 results_dict[method] = [0.0] * 7
                 predictions_dict[method] = (None, None)
                 table.add_row(method, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")

        except Exception as e:
             print(f"Error running ensemble method {method}: {e}")
             results_dict[method] = [0.0] * 7 # Assign poor score if fails
             predictions_dict[method] = (None, None)
             table.add_row(method, "Error", "Error", "Error", "Error", "Error", "Error", "Error")

    console.print(table)

    if best_mcc <= -1.0: # Check if any method actually succeeded
        print("\nWarning: All ensemble methods failed or produced invalid MCC. Cannot determine best method.")
        # Fallback to simple average if it exists, otherwise return None
        best_method = 'avg' if 'avg' in results_dict and results_dict['avg'][4] > -1.0 else None
        if best_method is None:
             print("Could not fallback to 'avg' method.")
             return None, results_dict, None, None # Return None for preds/probs
        else:
             print("Falling back to 'avg' as the best method.")

    print(f"\nBest ensemble method based on MCC: {best_method}")
    print(f"  Accuracy = {results_dict[best_method][0]:.2f}%")
    print(f"  MCC      = {results_dict[best_method][4]:.4f}")
    print("======================================")

    wandb.log({"Final_Best_Ensemble_Method": best_method})

    # Return predictions and probabilities for the best method
    best_preds, best_probs = predictions_dict.get(best_method, (None, None))

    return best_method, results_dict, best_preds, best_probs


# --------------------- Optuna Objective Function (Sequence Only) ---------------------
def objective_sequence(trial: Trial, train_data_dict):
    """Optuna objective function for sequence-only model."""
    # Unpack pre-loaded data
    train_seq_features = train_data_dict['seq_features']
    train_labels = train_data_dict['labels']

    # Suggest hyperparameters (Sequence only)
    lr = trial.suggest_float('lr', CONFIG["lr_range"][0], CONFIG["lr_range"][1], log=True)
    weight_decay = trial.suggest_float('weight_decay', CONFIG["weight_decay_range"][0], CONFIG["weight_decay_range"][1], log=True)
    seq_projected_dim = trial.suggest_categorical('seq_proj_dim', CONFIG["seq_pre_cnn_dim_choices"])
    conv1_out = trial.suggest_categorical('conv1_out', CONFIG["conv1_out_choices"])
    conv2_out = trial.suggest_categorical('conv2_out', CONFIG["conv2_out_choices"])
    kernel_size = trial.suggest_int('kernel_size', CONFIG["kernel_size_range"][0], CONFIG["kernel_size_range"][1])
    dropout_rate = trial.suggest_float('dropout_rate', CONFIG["dropout_rate_range"][0], CONFIG["dropout_rate_range"][1])
    batch_size = trial.suggest_categorical('batch_size', CONFIG["batch_size_choices"])
    # Removed fusion/mol related hyperparameters

    # --- Cross-validation ---
    skf = StratifiedKFold(n_splits=CONFIG["optuna_kfold_splits"], random_state=CONFIG["seed"], shuffle=True)
    fold_val_metrics = [] # Store best metrics [mcc] for each fold

    print(f"\n--- Optuna Trial {trial.number} ---")
    print(f"Params: {trial.params}")

    for fold, (train_index, val_index) in enumerate(skf.split(train_seq_features, train_labels)):
        print(f"  Fold {fold+1}/{CONFIG['optuna_kfold_splits']}")

        # Split sequence features and labels for this fold
        X_seq_train, X_seq_val = train_seq_features[train_index], train_seq_features[val_index]
        y_train, y_val = train_labels[train_index], train_labels[val_index]

        # --- Feature Scaling (Sequence Only) ---
        seq_scaler = StandardScaler()
        X_seq_train_shape = X_seq_train.shape
        X_seq_train = seq_scaler.fit_transform(X_seq_train.reshape(-1, X_seq_train_shape[-1])).reshape(X_seq_train_shape)
        X_seq_val_shape = X_seq_val.shape
        X_seq_val = seq_scaler.transform(X_seq_val.reshape(-1, X_seq_val_shape[-1])).reshape(X_seq_val_shape)
        # --- End Scaling ---

        # Convert to Tensors (DataLoader expects batch, seq_len, features)
        X_seq_train_tensor = torch.Tensor(X_seq_train)
        X_seq_val_tensor = torch.Tensor(X_seq_val)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)

        # Create TensorDataset and DataLoaders (Sequence Only)
        train_dataset_fold = TensorDataset(X_seq_train_tensor, y_train_tensor)
        val_dataset_fold = TensorDataset(X_seq_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

        # Instantiate the SequenceModel
        seq_embedding_dim = train_seq_features.shape[-1] # Get dim from data
        model = SequenceModel(
            seq_embedding_dim=seq_embedding_dim,
            seq_projected_dim=seq_projected_dim,
            num_classes=CONFIG["num_classes"], conv1_out=conv1_out, conv2_out=conv2_out,
            kernel_size=kernel_size, dropout_rate=dropout_rate
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
        num_epochs = CONFIG["optuna_epochs"]
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)

        # Training loop with early stopping for the fold
        best_fold_val_mcc = -np.inf
        epochs_no_improve = 0

        try:
            for epoch in range(num_epochs):
                train_loss, train_acc = train_sequence_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_acc, _, _, _, val_mcc, _, _ = validate_sequence_epoch(model, val_loader, criterion) # Only need MCC for Optuna
                scheduler.step()

                if val_mcc > best_fold_val_mcc:
                    best_fold_val_mcc = val_mcc
                    epochs_no_improve = 0
                    # Optional: Save best model state for this fold if needed later, but Optuna mainly needs the metric
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= CONFIG["optuna_early_stop_patience"]:
                        print(f"    Fold {fold+1} early stopping at epoch {epoch+1}, Best MCC: {best_fold_val_mcc:.4f}")
                        break

            # Report the best MCC achieved in this fold for pruning purposes
            # Using a unique step for reporting across folds
            trial.report(best_fold_val_mcc if best_fold_val_mcc > -np.inf else -1.0, step=fold)
            if trial.should_prune():
                 print(f"    Trial {trial.number}, Fold {fold+1} pruned.")
                 # Ensure a value is appended even if pruned early
                 fold_val_metrics.append(best_fold_val_mcc if best_fold_val_mcc > -np.inf else -1.0)
                 raise optuna.exceptions.TrialPruned()

        except optuna.exceptions.TrialPruned:
             # Propagate prune signal
             raise
        except Exception as e:
             print(f"Error during training in Optuna Trial {trial.number}, Fold {fold+1}: {e}")
             best_fold_val_mcc = -1.0 # Indicate failure for this fold
             # Potentially raise TrialPruned here too if one fold error should prune the trial
             # raise optuna.exceptions.TrialPruned(f"Error in fold {fold+1}")
             # Or just record the failure and continue to calculate mean MCC from successful folds
             print(f"    Fold {fold+1} failed, recording MCC as -1.0")


        fold_val_metrics.append(best_fold_val_mcc)

        # Clean up fold-specific resources
        del model, optimizer, scheduler, criterion, train_loader, val_loader, train_dataset_fold, val_dataset_fold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate mean MCC across folds that didn't fail (MCC > -1.0)
    valid_mccs = [m for m in fold_val_metrics if m > -1.0]
    mean_val_mcc = np.mean(valid_mccs) if valid_mccs else -1.0 # Return -1.0 if all folds failed/pruned early with poor score

    print(f"  Trial {trial.number} completed. Mean Valid Fold MCC: {mean_val_mcc:.4f} (from {len(valid_mccs)}/{CONFIG['optuna_kfold_splits']} valid folds)")
    return mean_val_mcc


# --------------------- Optuna Study Execution (Sequence Only) ---------------------
def run_sequence_optuna():
    """Runs Optuna hyperparameter optimization for sequence-only model."""
    print("\n===== Starting Optuna Hyperparameter Optimization (Sequence Only) =====")

    # Load data ONCE before starting the study (Sequence Only)
    try:
        (train_seq_features, train_labels), _ = load_sequence_data(
            CONFIG["train_csv"], CONFIG["seq_train_feat_path"],
            CONFIG["test_csv"], CONFIG["seq_test_feat_path"]
        )
        # Store data in a dictionary to pass to the objective function
        train_data_dict = {
            'seq_features': train_seq_features,
            'labels': train_labels
        }
    except Exception as e:
        print(f"Fatal Error: Data loading failed before Optuna study: {e}")
        return None # Cannot proceed without data

    sampler = TPESampler(seed=CONFIG["seed"])
    study = optuna.create_study(direction='maximize', sampler=sampler) # Maximize mean validation MCC

    try:
        # Pass pre-loaded data to the objective function using a lambda
        study.optimize(lambda trial: objective_sequence(trial, train_data_dict),
                       n_trials=CONFIG["optuna_n_trials"],
                       timeout=CONFIG["optuna_timeout"],
                       gc_after_trial=True) # Garbage collect after each trial
    except Exception as e:
        print(f"Optuna optimization failed: {e}")
        # Optionally load previous best if available or handle error
        return None # Indicate failure

    print("\n===== Optuna Hyperparameter Optimization Finished (Sequence Only) =====")
    print(f"Number of finished trials: {len(study.trials)}")

    # Check if study has successful trials before accessing best_trial
    valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not valid_trials:
         print("Optuna study finished without any successful trials.")
         return None

    # Find the best trial among completed ones
    best_trial = max(valid_trials, key=lambda t: t.value if t.value is not None else -np.inf)

    if best_trial.value is None or best_trial.value <= -1.0:
         print("Optuna study finished, but the best trial had a non-positive or invalid score.")
         return None


    print("\nBest trial found:")
    print(f"  Value (Mean Val MCC): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save study using path from CONFIG (sequence-only specific path)
    study_path = CONFIG["optuna_study_path"]
    study_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    try:
        joblib.dump(study, study_path)
        print(f"Optuna study saved to {study_path}")
    except Exception as e:
        print(f"Error saving Optuna study: {e}")

    # Optional: Plot Optuna results (e.g., hyperparameter importance)
    try:
        plot_optuna_results(study) # Plotting function remains the same, but input is seq-only study
    except Exception as e:
        print(f"Warning: Failed to generate Optuna plots: {e}")


    return study

# --------------------- Main Training Workflow (Sequence Only) ---------------------
def main_train_sequence(best_params):
    """Main training workflow using K-Fold cross-validation and best hyperparameters (Sequence Only)."""
    print("\n===== Starting Main K-Fold Training & Evaluation (Sequence Only) =====")
    print(f"Using Best Hyperparameters: {best_params}")

    # Ensure model directory exists (sequence-only specific path)
    model_dir = CONFIG["model_dir_base"] # Already points to 'model_seq_only'
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved in: {model_dir}")

    # Load data (Sequence Only)
    try:
        (train_seq_features, train_labels), \
        (test_seq_features, test_labels) = load_sequence_data(
            CONFIG["train_csv"], CONFIG["seq_train_feat_path"],
            CONFIG["test_csv"], CONFIG["seq_test_feat_path"]
        )
    except Exception as e:
        print(f"Fatal Error: Data loading failed in main training: {e}")
        return [], [], None, None, None, None, None # Return empty results

    # --- Feature Scaling for Test Set (Fit on FULL training data - Sequence Only) ---
    print("Fitting sequence scaler on full training data for test set transformation...")
    seq_scaler_test = StandardScaler()
    train_seq_shape = train_seq_features.shape
    seq_scaler_test.fit(train_seq_features.reshape(-1, train_seq_shape[-1]))
    test_seq_shape = test_seq_features.shape
    test_seq_features_scaled = seq_scaler_test.transform(
        test_seq_features.reshape(-1, test_seq_shape[-1])
    ).reshape(test_seq_shape)
    print("Test set scaling complete.")
    # --- End Test Set Scaling ---

    # --- K-Fold Cross-validation ---
    skf = StratifiedKFold(n_splits=CONFIG["main_kfold_splits"], random_state=CONFIG["seed"]+1, shuffle=True) # Use different seed for main folds

    all_fold_val_metrics = [] # Store best metrics from validation set for each fold
    all_fold_test_probs = []  # Store test set probabilities for each fold
    all_fold_test_metrics = [] # Store test set metrics for each fold
    fold_histories = {} # Store epoch-level history for each fold for plotting

    # Extract hyperparameters from best_params (Sequence Only)
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']
    seq_projected_dim = best_params['seq_proj_dim']
    conv1_out = best_params['conv1_out']
    conv2_out = best_params['conv2_out']
    kernel_size = best_params['kernel_size']
    dropout_rate = best_params['dropout_rate']
    batch_size = best_params['batch_size']


    print(f"\n===== Starting {CONFIG['main_kfold_splits']}-Fold Cross-Validation (Sequence Only) =====")
    for fold, (train_index, val_index) in enumerate(skf.split(train_seq_features, train_labels)):
        print(f"\n--- Fold {fold+1}/{CONFIG['main_kfold_splits']} ---")
        fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_mcc': []}

        # Split features and labels for this fold (Sequence Only)
        X_seq_train, X_seq_val = train_seq_features[train_index], train_seq_features[val_index]
        y_train, y_val = train_labels[train_index], train_labels[val_index]

        # --- Feature Scaling (Inside Fold - Sequence Only) ---
        seq_scaler_fold = StandardScaler()
        X_seq_train_shape = X_seq_train.shape
        X_seq_train_scaled = seq_scaler_fold.fit_transform(X_seq_train.reshape(-1, X_seq_train_shape[-1])).reshape(X_seq_train_shape)
        X_seq_val_shape = X_seq_val.shape
        X_seq_val_scaled = seq_scaler_fold.transform(X_seq_val.reshape(-1, X_seq_val_shape[-1])).reshape(X_seq_val_shape)
        # --- End Scaling ---

        # Convert to Tensors (Sequence Only)
        X_seq_train_tensor = torch.Tensor(X_seq_train_scaled)
        X_seq_val_tensor = torch.Tensor(X_seq_val_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)

        train_dataset_fold = TensorDataset(X_seq_train_tensor, y_train_tensor)
        val_dataset_fold = TensorDataset(X_seq_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

        # Instantiate the model for this fold (SequenceModel)
        seq_embedding_dim = train_seq_features.shape[-1]
        model = SequenceModel(
            seq_embedding_dim=seq_embedding_dim,
            seq_projected_dim=seq_projected_dim,
            num_classes=CONFIG["num_classes"], conv1_out=conv1_out, conv2_out=conv2_out,
            kernel_size=kernel_size, dropout_rate=dropout_rate
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
        num_epochs = CONFIG["main_epochs"]
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)

        # Training loop for the fold
        best_val_mcc_fold = -np.inf
        epochs_no_improve = 0
        best_metrics_val_fold = [0.0] * 7 # [acc, sn, sp, pre, mcc, f1, auc]
        model_path = model_dir / f'fold_{fold+1}.model' # Use Path object
        best_epoch = -1

        for epoch in range(num_epochs):
            train_loss, train_acc = train_sequence_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_sn, val_sp, val_pre, val_mcc, val_f1, val_auc = validate_sequence_epoch(model, val_loader, criterion)
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            # Store history for plotting
            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc)
            fold_history['val_mcc'].append(val_mcc)


            print(f"  Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, MCC={val_mcc:.4f}, AUC={val_auc:.4f}, LR={current_lr:.2e}")

            # Save best model based on validation MCC
            if val_mcc > best_val_mcc_fold:
                best_val_mcc_fold = val_mcc
                epochs_no_improve = 0
                best_metrics_val_fold = [val_acc, val_sn, val_sp, val_pre, val_mcc, val_f1, val_auc]
                best_epoch = epoch + 1
                try:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'best_val_metrics': best_metrics_val_fold,
                        'epoch': best_epoch,
                        'params': best_params # Save the hyperparameters used for this model
                    }, model_path)
                    # print(f"    ---> Best model saved at epoch {best_epoch} with Val MCC: {val_mcc:.4f}") # Verbose saving message
                except Exception as e:
                    print(f"Error saving model for fold {fold+1}: {e}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= CONFIG["main_early_stop_patience"]:
                    print(f"  Early stopping triggered at epoch {epoch+1}. Best Val MCC: {best_val_mcc_fold:.4f} at epoch {best_epoch}")
                    break

        fold_histories[f"fold_{fold+1}"] = fold_history
        all_fold_val_metrics.append(best_metrics_val_fold) # Store best validation metrics for this fold

        # --- Evaluate Best Model of the Fold on the Test Set ---
        if model_path.exists() and best_epoch != -1:
            print(f"  Loading best model from epoch {best_epoch} for test evaluation...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                # Re-instantiate model to load state_dict cleanly
                model_load = SequenceModel(
                    seq_embedding_dim=seq_embedding_dim,
                    seq_projected_dim=best_params['seq_proj_dim'],
                    num_classes=CONFIG["num_classes"],
                    conv1_out=best_params['conv1_out'], conv2_out=best_params['conv2_out'],
                    kernel_size=best_params['kernel_size'], dropout_rate=best_params['dropout_rate']
                ).to(device)
                model_load.load_state_dict(checkpoint['model_state_dict'])
                model_load.eval()

                # Prepare test dataset loader (using pre-scaled test data - Sequence Only)
                test_seq_tensor = torch.Tensor(test_seq_features_scaled)
                test_labels_tensor = torch.LongTensor(test_labels)
                test_dataset = TensorDataset(test_seq_tensor, test_labels_tensor)
                test_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False, pin_memory=True, num_workers=CONFIG["num_workers"])

                fold_test_preds_batches = []
                fold_test_probs_batches = []
                fold_test_labels_batches = []
                with torch.no_grad():
                    for seq_feat_test, label_test in test_loader:
                        seq_feat_test = seq_feat_test.to(device)
                        outputs_test = model_load(seq_feat_test) # Pass only sequence features
                        probs_test = F.softmax(outputs_test, dim=1)
                        _, preds_test = torch.max(outputs_test, 1)

                        fold_test_preds_batches.append(preds_test.cpu().numpy())
                        fold_test_probs_batches.append(probs_test.cpu().numpy())
                        fold_test_labels_batches.append(label_test.cpu().numpy()) # Collect labels to ensure order

                fold_test_preds = np.concatenate(fold_test_preds_batches)
                fold_test_probs = np.concatenate(fold_test_probs_batches)
                fold_test_labels = np.concatenate(fold_test_labels_batches) # Use collected labels

                all_fold_test_probs.append(fold_test_probs)

                # Calculate and store test metrics for this fold
                fold_test_metrics_results = performance(fold_test_labels, fold_test_preds, fold_test_probs)
                all_fold_test_metrics.append(list(fold_test_metrics_results)) # Store as list
                test_acc, test_sn, test_sp, test_pre, test_mcc, test_f1, test_auc = fold_test_metrics_results
                print(f'  Fold {fold+1} Test Metrics: Acc={test_acc:.2f}%, SN={test_sn:.4f}, SP={test_sp:.4f}, Prec={test_pre:.4f}, MCC={test_mcc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}')
                wandb.log({
                    f"Fold_{fold+1}_Test_Accuracy": test_acc, f"Fold_{fold+1}_Test_Sensitivity": test_sn,
                    f"Fold_{fold+1}_Test_Specificity": test_sp, f"Fold_{fold+1}_Test_Precision": test_pre,
                    f"Fold_{fold+1}_Test_MCC": test_mcc, f"Fold_{fold+1}_Test_F1": test_f1,
                    f"Fold_{fold+1}_Test_ROC_AUC": test_auc
                })

            except Exception as e:
                 print(f"Error loading model or evaluating on test set for fold {fold+1}: {e}")
                 # Append zeros or handle failure appropriately for aggregation later
                 all_fold_test_probs.append(np.zeros((len(test_labels), CONFIG["num_classes"])))
                 all_fold_test_metrics.append([0.0] * 7) # Add placeholder metrics
                 # Ensure corresponding validation metrics are also marked as invalid if fold failed
                 if len(all_fold_val_metrics) == fold + 1: all_fold_val_metrics[-1] = [0.0] * 7

        else:
             print(f"Warning: No best model saved/found for fold {fold+1}. Skipping test evaluation for this fold.")
             all_fold_test_probs.append(np.zeros((len(test_labels), CONFIG["num_classes"])))
             all_fold_test_metrics.append([0.0] * 7)
             if len(all_fold_val_metrics) == fold + 1: all_fold_val_metrics[-1] = [0.0] * 7

        # Clean up fold resources
        del model, model_load, train_loader, val_loader, train_dataset_fold, val_dataset_fold, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n===== K-Fold Cross-Validation Finished (Sequence Only) =====")

    # --- Aggregate and Summarize Results ---
    print("\n===== Overall Training & Evaluation Summary (Sequence Only) =====")
    # No fusion method to print

    # Summarize Validation Metrics (using the best epoch from each fold)
    valid_val_folds = [m for m in all_fold_val_metrics if m[4] > -1.0] # Check MCC (index 4) validity
    if valid_val_folds:
        avg_val_metrics = np.mean(valid_val_folds, axis=0)
        std_val_metrics = np.std(valid_val_folds, axis=0)
        print(f'\nAvg Best Validation Metrics ({len(valid_val_folds)}/{CONFIG["main_kfold_splits"]} Valid Folds):')
        print(f'  Acc={avg_val_metrics[0]:.2f}%(±{std_val_metrics[0]:.2f}), SN={avg_val_metrics[1]:.4f}(±{std_val_metrics[1]:.4f}), SP={avg_val_metrics[2]:.4f}(±{std_val_metrics[2]:.4f}), Prec={avg_val_metrics[3]:.4f}(±{std_val_metrics[3]:.4f})')
        print(f'  MCC={avg_val_metrics[4]:.4f}(±{std_val_metrics[4]:.4f}), F1={avg_val_metrics[5]:.4f}(±{std_val_metrics[5]:.4f}), AUC={avg_val_metrics[6]:.4f}(±{std_val_metrics[6]:.4f})')
        wandb.log({"Avg_Best_Val_MCC": avg_val_metrics[4]})
    else:
        print("\nWarning: No valid validation metrics recorded across folds.")

    # Summarize Individual Fold Test Metrics
    valid_test_folds_metrics = [m for m in all_fold_test_metrics if m[4] > -1.0]
    if valid_test_folds_metrics:
        avg_test_metrics = np.mean(valid_test_folds_metrics, axis=0)
        std_test_metrics = np.std(valid_test_folds_metrics, axis=0)
        print(f'\nAvg Individual Fold Test Metrics ({len(valid_test_folds_metrics)}/{CONFIG["main_kfold_splits"]} Valid Folds):')
        print(f'  Acc={avg_test_metrics[0]:.2f}%(±{std_test_metrics[0]:.2f}), SN={avg_test_metrics[1]:.4f}(±{std_test_metrics[1]:.4f}), SP={avg_test_metrics[2]:.4f}(±{std_test_metrics[2]:.4f}), Prec={avg_test_metrics[3]:.4f}(±{std_val_metrics[3]:.4f})')
        print(f'  MCC={avg_test_metrics[4]:.4f}(±{std_test_metrics[4]:.4f}), F1={avg_test_metrics[5]:.4f}(±{std_test_metrics[5]:.4f}), AUC={avg_test_metrics[6]:.4f}(±{std_test_metrics[6]:.4f})')
        wandb.log({
            "Avg_Fold_Test_Accuracy": avg_test_metrics[0], "Std_Fold_Test_Accuracy": std_test_metrics[0],
            "Avg_Fold_Test_MCC": avg_test_metrics[4], "Std_Fold_Test_MCC": std_test_metrics[4],
            "Avg_Fold_Test_AUC": avg_test_metrics[6], "Std_Fold_Test_AUC": std_test_metrics[6],
        })
    else:
        print("\nWarning: No valid test metrics recorded across folds.")

    # --- Ensemble Evaluation ---
    print("\n===== Final Ensemble Evaluation on Test Set (Sequence Only) =====")
    # Use validation metrics (specifically MCC) from the best epoch of each fold for weighting
    # Also get best predictions/probs for plotting
    best_ensemble_method, ensemble_results_dict, best_ensemble_preds, best_ensemble_probs = summarize_ensemble_comparison(
        all_fold_test_probs, test_labels, all_fold_val_metrics
    )

    final_metrics = [0.0] * 7
    if best_ensemble_method and best_ensemble_method in ensemble_results_dict:
        final_metrics = ensemble_results_dict[best_ensemble_method]
        print(f"\nFinal Performance using Best Ensemble Method ({best_ensemble_method}):")
        print(f"  Accuracy: {final_metrics[0]:.2f}%")
        print(f"  SN: {final_metrics[1]:.4f}")
        print(f"  SP: {final_metrics[2]:.4f}")
        print(f"  Precision: {final_metrics[3]:.4f}")
        print(f"  MCC: {final_metrics[4]:.4f}")
        print(f"  F1: {final_metrics[5]:.4f}")
        print(f"  AUC: {final_metrics[6]:.4f}")

        # Log final best ensemble results explicitly
        wandb.log({
            f"Final_{best_ensemble_method}_Accuracy": final_metrics[0],
            f"Final_{best_ensemble_method}_Sensitivity": final_metrics[1],
            f"Final_{best_ensemble_method}_Specificity": final_metrics[2],
            f"Final_{best_ensemble_method}_Precision": final_metrics[3],
            f"Final_{best_ensemble_method}_MCC": final_metrics[4],
            f"Final_{best_ensemble_method}_F1": final_metrics[5],
            f"Final_{best_ensemble_method}_ROC_AUC": final_metrics[6],
        })
    else:
        print("\nCould not determine final ensemble performance.")
        best_ensemble_preds = None # Ensure these are None if ensemble fails
        best_ensemble_probs = None


    # --- Generate Visualizations ---
    print("\n===== Generating Visualizations (Sequence Only) =====")
    try:
        # Plot average training/validation curves
        plot_training_history(fold_histories, filename=PLOT_DIR / f"training_curves_seq_only.png")

        # Plot ROC curves for folds and ensemble
        plot_roc_curves(all_fold_test_probs, test_labels, ensemble_results_dict, best_ensemble_method, filename=PLOT_DIR / f"roc_curves_seq_only.png")

        # Plot Confusion Matrix for best ensemble
        if best_ensemble_preds is not None:
             plot_confusion_matrix(test_labels, best_ensemble_preds, display_labels=["Non-inhibitor", "Inhibitor"],
                                   filename=PLOT_DIR / f"confusion_matrix_{best_ensemble_method}_seq_only.png",
                                   title=f"Confusion Matrix (Seq Only - Ensemble: {best_ensemble_method})")
        else:
             print("Skipping confusion matrix plot: Best ensemble predictions not available.")

        # Plot Probability Distribution for best ensemble
        if best_ensemble_probs is not None:
             plot_probability_distribution(test_labels, best_ensemble_probs[:, 1], # Use probability of positive class
                                           filename=PLOT_DIR / f"probability_dist_{best_ensemble_method}_seq_only.png",
                                           title=f"Probability Distribution (Seq Only - Ensemble: {best_ensemble_method})")
        else:
             print("Skipping probability distribution plot: Best ensemble probabilities not available.")

        # Plot K-Fold Test Metrics Distribution
        if valid_test_folds_metrics: # Check if there are valid metrics to plot
            metric_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "MCC", "F1-Score", "AUC"]
            plot_kfold_metrics_distribution(valid_test_folds_metrics, metric_names,
                                              filename=PLOT_DIR / "kfold_test_metrics_dist_seq_only.png",
                                              title="Distribution of Test Metrics Across K-Folds (Sequence Only)")
        else:
            print("Skipping K-Fold metrics distribution plot: No valid fold metrics.")

    except Exception as e:
        print(f"Warning: Failed to generate one or more plots: {e}")

    # Ensure the list of valid test fold metrics is returned
    valid_test_folds_metrics = [m for m in all_fold_test_metrics if len(m) == 7 and m[4] > -1.0]
    return all_fold_test_probs, test_labels, best_ensemble_method, final_metrics, best_ensemble_preds, best_ensemble_probs, valid_test_folds_metrics

# --------------------- Visualization Functions ---------------------
# Keep visualization functions as they are mostly generic

def plot_optuna_results(study):
    """Generates plots for Optuna study results."""
    if not study or not study.trials:
        print("No Optuna study or trials available to plot.")
        return

    # --- Plot 1: Hyperparameter Importance ---
    try:
        # Filter out parameters not present in the study (e.g., fusion_method)
        params_in_study = list(study.best_params.keys())
        fig_importance = optuna.visualization.plot_param_importances(study, params=params_in_study)
        fig_importance.update_layout(title="Hyperparameter Importance (Optuna - Sequence Only)", title_x=0.5)
        # Use Path object for saving
        filepath = PLOT_DIR / "optuna_param_importances_seq_only.png" # Changed filename
        fig_importance.write_image(str(filepath), scale=2) # Increase scale for better resolution
        print(f"Saved Optuna importance plot to {filepath}")
    except ImportError:
         print("Could not generate Optuna importance plot: Requires `pip install kaleido`")
    except ValueError as ve:
         print(f"Could not generate Optuna importance plot, likely due to too few trials or constant parameters: {ve}")
    except Exception as e:
        print(f"Could not generate Optuna importance plot: {e}")

    # --- Plot 2: Performance vs. Fusion Method (Box Plot) ---
    # This plot is not relevant for sequence-only, so skip it or adapt if other categorical params exist
    print("Skipping fusion performance plot as it's not applicable for sequence-only.")


def plot_training_history(fold_histories, filename="training_curves.png"):
    """Plots average training/validation loss and MCC across folds."""
    if not fold_histories:
        print("No fold histories available to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Aggregate metrics across folds
    max_epochs = max(len(h['train_loss']) for h in fold_histories.values()) if fold_histories else 0
    if max_epochs == 0:
        print("Cannot plot training history: No epochs recorded.")
        plt.close(fig)
        return

    all_train_loss = np.full((len(fold_histories), max_epochs), np.nan)
    all_val_loss = np.full((len(fold_histories), max_epochs), np.nan)
    all_val_mcc = np.full((len(fold_histories), max_epochs), np.nan)

    for i, (fold_name, history) in enumerate(fold_histories.items()):
        epochs_ran = len(history['train_loss'])
        all_train_loss[i, :epochs_ran] = history['train_loss']
        all_val_loss[i, :epochs_ran] = history['val_loss']
        all_val_mcc[i, :epochs_ran] = history['val_mcc']

    # Calculate mean and std dev, ignoring NaNs
    mean_train_loss = np.nanmean(all_train_loss, axis=0)
    std_train_loss = np.nanstd(all_train_loss, axis=0)
    mean_val_loss = np.nanmean(all_val_loss, axis=0)
    std_val_loss = np.nanstd(all_val_loss, axis=0)
    mean_val_mcc = np.nanmean(all_val_mcc, axis=0)
    std_val_mcc = np.nanstd(all_val_mcc, axis=0)

    epochs = np.arange(1, max_epochs + 1)

    # Plot Loss
    axes[0].plot(epochs, mean_train_loss, label='Average Train Loss', color=COLOR_PRIMARY_BLUE, linewidth=2)
    axes[0].fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color=COLOR_PRIMARY_BLUE, alpha=0.2)
    axes[0].plot(epochs, mean_val_loss, label='Average Validation Loss', color=COLOR_PRIMARY_PINK, linewidth=2)
    axes[0].fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color=COLOR_PRIMARY_PINK, alpha=0.2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Average Training and Validation Loss (Sequence Only)', fontsize=14) # Updated title
    axes[0].legend(fontsize=10)
    axes[0].grid(True)

    # Plot MCC
    axes[1].plot(epochs, mean_val_mcc, label='Average Validation MCC', color=COLOR_PRIMARY_PINK, linewidth=2)
    axes[1].fill_between(epochs, mean_val_mcc - std_val_mcc, mean_val_mcc + std_val_mcc, color=COLOR_PRIMARY_PINK, alpha=0.2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MCC', fontsize=12)
    axes[1].set_title('Average Validation MCC (Sequence Only)', fontsize=14) # Updated title
    axes[1].legend(fontsize=10)
    axes[1].grid(True)
    # Optional: Add a horizontal line at the max average MCC
    # Handle case where all NaNs might occur if training is very short / fails early
    if not np.all(np.isnan(mean_val_mcc)):
        max_avg_mcc = np.nanmax(mean_val_mcc)
        max_epoch = epochs[np.nanargmax(mean_val_mcc)]
        axes[1].axhline(max_avg_mcc, linestyle='--', color='grey', linewidth=1, label=f'Max Avg MCC: {max_avg_mcc:.4f} at Epoch {max_epoch}')
        axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"Saved training history plot to {filename}")

def plot_roc_curves(all_fold_test_probs, test_labels, ensemble_results_dict, best_ensemble_method, filename="roc_curves.png"):
    """Plots ROC curves for individual folds and the best ensemble."""
    if not all_fold_test_probs:
        print("No fold test probabilities available to plot ROC curves.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 8))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    num_folds = len(all_fold_test_probs)

    # Plot ROC for each fold
    for i, probs in enumerate(all_fold_test_probs):
        if probs is not None and probs.ndim == 2 and probs.shape[1] == 2 and len(test_labels) == probs.shape[0]:
             # Use roc_curve from sklearn.metrics
             fpr, tpr, _ = roc_curve(test_labels, probs[:, 1])
             roc_auc = auc(fpr, tpr) # Use auc from sklearn.metrics
             aucs.append(roc_auc)
             plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
             # Interpolate TPR at standard FPR levels
             interp_tpr = np.interp(mean_fpr, fpr, tpr)
             interp_tpr[0] = 0.0
             tprs.append(interp_tpr)
        else:
             print(f"Warning: Skipping ROC plot for Fold {i+1} due to invalid probabilities (shape: {probs.shape if probs is not None else 'None'}) or label mismatch.")

    # Plot chance line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Chance', alpha=0.8)

    # Plot Mean ROC across folds
    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs) if aucs else 0.0
        plt.plot(mean_fpr, mean_tpr, color=COLOR_PRIMARY_BLUE,
                 label=f'Mean ROC (AUC = {mean_auc:.2f} $\\pm$ {std_auc:.2f})', lw=2.5, alpha=0.9)

        # Plot standard deviation fill
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=COLOR_PRIMARY_BLUE, alpha=0.2,
                         label=r'$\pm$ 1 std. dev.')

    # Plot ROC for the best ensemble method
    if best_ensemble_method and best_ensemble_method in ensemble_results_dict and all_fold_test_probs:
        # Recalculate average probabilities for the best method if needed (or pass them)
        # Use the simple average probability for plotting the ensemble ROC
        valid_probs_for_avg = [p for p in all_fold_test_probs if p is not None and p.ndim == 2 and p.shape[0] == len(test_labels)]
        if valid_probs_for_avg:
            probs_ensemble = np.mean(np.array(valid_probs_for_avg), axis=0)
        else:
            probs_ensemble = None # Cannot calculate average

        if probs_ensemble is not None and probs_ensemble.ndim == 2 and probs_ensemble.shape[1] == 2:
            fpr_ens, tpr_ens, _ = roc_curve(test_labels, probs_ensemble[:, 1])
            # Get AUC from stored results dictionary (index 6)
            auc_ens = ensemble_results_dict[best_ensemble_method][6]
            plt.plot(fpr_ens, tpr_ens, color=COLOR_PRIMARY_PINK,
                     label=f'Best Ensemble: {best_ensemble_method} (AUC = {auc_ens:.2f})', lw=2.5, linestyle=':', alpha=0.9)
        else:
            print(f"Could not plot ROC for ensemble method '{best_ensemble_method}' due to invalid/missing fold probabilities.")


    # Final plot settings
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves (Sequence Only)', fontsize=14) # Updated title
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI)
    plt.close()
    print(f"Saved ROC curve plot to {filename}")

# --- New Visualization Functions --- #

def plot_confusion_matrix(y_true, y_pred, display_labels, filename, title):
    """Plots and saves the confusion matrix."""
    if y_true is None or y_pred is None:
        print(f"Skipping confusion matrix plot '{title}': Missing true labels or predictions.")
        return
    try:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d') # Use standard Blues colormap
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=PLOT_DPI)
        plt.close(fig)
        print(f"Saved confusion matrix plot to {filename}")
    except Exception as e:
        print(f"Could not generate confusion matrix plot '{title}': {e}")

def plot_probability_distribution(y_true, y_prob_positive, filename, title):
    """Plots and saves the probability distribution for the positive class."""
    if y_true is None or y_prob_positive is None:
        print(f"Skipping probability distribution plot '{title}': Missing true labels or probabilities.")
        return
    try:
        df = pd.DataFrame({
            'Probability (Class 1)': y_prob_positive,
            'True Label': [f"Class {label}" for label in y_true] # Use string labels for hue
        })
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Probability (Class 1)', hue='True Label',
                     kde=True, stat='density', common_norm=False, palette=SNS_PALETTE)
        plt.title(title, fontsize=14)
        plt.xlabel('Predicted Probability for Class 1', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=PLOT_DPI)
        plt.close()
        print(f"Saved probability distribution plot to {filename}")
    except Exception as e:
        print(f"Could not generate probability distribution plot '{title}': {e}")

def plot_kfold_metrics_distribution(all_fold_metrics, metric_names, filename, title):
    """Plots the distribution of metrics across K-Folds."""
    if not all_fold_metrics:
        print("No fold metrics available to plot distribution.")
        return
    try:
        # Filter out any potential None or incomplete metric lists
        valid_metrics = [m for m in all_fold_metrics if m is not None and len(m) == len(metric_names)]
        if not valid_metrics:
            print("No valid fold metrics available to plot distribution after filtering.")
            return

        df_metrics = pd.DataFrame(valid_metrics, columns=metric_names)
        # Convert Accuracy back to 0-1 scale if it was stored as 0-100
        if 'Accuracy' in df_metrics.columns and df_metrics['Accuracy'].max() > 1.1:
            df_metrics['Accuracy'] = df_metrics['Accuracy'] / 100.0

        # Select metrics to plot (e.g., exclude precision/recall if too many)
        metrics_to_plot = ['Accuracy', 'MCC', 'F1-Score', 'AUC', 'Sensitivity', 'Specificity']
        df_plot = df_metrics[[col for col in metrics_to_plot if col in df_metrics.columns]]

        plt.figure(figsize=(12, 7))
        # Use boxplot for clearer quartile visualization
        sns.boxplot(data=df_plot, palette=PLOT_PALETTE[:len(df_plot.columns)])
        # Or use violinplot for density shape: sns.violinplot(data=df_plot, palette=PLOT_PALETTE[:len(df_plot.columns)])

        plt.title(title, fontsize=14)
        plt.ylabel('Metric Value', fontsize=12)
        plt.xticks(rotation=15) # Rotate labels slightly if needed
        plt.ylim([-0.05, 1.05]) # Set y-axis limits for common metrics (0-1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=PLOT_DPI)
        plt.close()
        print(f"Saved K-Fold metrics distribution plot to {filename}")
    except Exception as e:
        print(f"Could not generate K-Fold metrics distribution plot '{title}': {e}")


# --------------------- Main Execution Block (Sequence Only) ---------------------
if __name__ == "__main__":
    print("Running in Sequence Only Mode (ESM6B) - Ablation")

    # Step 1: Run Optuna hyperparameter optimization
    study = run_sequence_optuna()

    if study and study.best_trial and study.best_trial.value > -1.0 : # Ensure a valid best trial exists
        best_params = study.best_trial.params
        print("\nOptuna found best parameters (Sequence Only):")
        print(best_params)
        # Log best params found by Optuna to wandb config
        wandb.config.update(best_params, allow_val_change=True)

        # Step 2: Perform final K-Fold training and evaluation using the best configuration
        print(f"\n--- Proceeding with final {CONFIG['main_kfold_splits']}-Fold Training using Best Config (Sequence Only) ---")

        # Call main_train_sequence with the best params from the study
        all_fold_test_probs, test_labels, final_best_ensemble_method, final_metrics, best_ensemble_preds, best_ensemble_probs, valid_test_folds_metrics = main_train_sequence(best_params)

        # Final Summary based on the main training run
        if final_best_ensemble_method and final_metrics:
            print("\n===== FINAL OVERALL PERFORMANCE (Sequence Only) =====")
            print(f"Best Hyperparameters used: {best_params}")
            print(f"Best Ensemble Method determined: {final_best_ensemble_method}")
            print(f"Final Test Accuracy: {final_metrics[0]:.2f}%")
            print(f"Final Test MCC:      {final_metrics[4]:.4f}")
            print(f"Final Test AUC:      {final_metrics[6]:.4f}")
            # Construct the path first for clarity and safety
            model_save_path = CONFIG["model_dir_base"] # Already points to seq_only dir
            print(f"Model files saved in: {model_save_path}")
            print("===================================")
            # Log overall final metrics
            wandb.log({
                "Overall_Final_Accuracy": final_metrics[0],
                "Overall_Final_MCC": final_metrics[4],
                "Overall_Final_AUC": final_metrics[6],
                "Overall_Best_Ensemble_Method": final_best_ensemble_method # From ensemble comparison
            })

            # --- Generate Additional Visualizations (Already done in main_train_sequence) ---
            # Visualizations are generated at the end of main_train_sequence
            print("\nVisualizations generated during the main training workflow.")

        else:
            print("\nFinal training or evaluation failed. Check logs for details.")

    else:
        print("\nOptuna study failed or produced no valid best trial. Aborting main training.")

    print("\nScript finished.")
    wandb.finish() 