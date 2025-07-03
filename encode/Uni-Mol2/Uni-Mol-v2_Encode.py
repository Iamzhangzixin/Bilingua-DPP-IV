"""
Uni-Mol Feature Extraction and Normalization Script
Improvements:
    1. Normalization:
        Uses MinMaxScaler for column-wise normalization, scaling feature values to [0, 1] range.
    2. Save paths:
        Normalized features saved to ./unimol_features/normalized
        Raw features saved to ./unimol_features/raw
    3. Flexible representation type selection:
        Can choose to extract molecular-level representations, atomic-level representations, or both.
    4. Atomic representation processing:
        Provides multiple ways to handle inconsistent atom counts across molecules, including padding, average pooling, and max pooling.
"""

# huggingface.co is not accessible, use hf-mirror.com mirror
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'    # Set Hugging Face mirror endpoint

from unimol_tools import UniMolRepr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import logging
import torch  # Import torch to check GPU
import pickle

# --- Global Configuration ---
# Set whether to perform normalization
USE_NORMALIZATION = False  # Set to True for column-wise normalization, False for no normalization

# Set whether to extract molecular-level and atomic-level representations
EXTRACT_MOLECULAR_REPR = False  # Set to True to extract molecular-level representations
EXTRACT_ATOMIC_REPR = True  # Set to True to extract atomic-level representations

# Set atomic-level representation processing method
# Options: 'raw' (original irregular shape, saved with pickle), 'pad' (zero padding), 'mean_pool' (average pooling), 'max_pool' (max pooling)
ATOMIC_REPR_PROCESS = 'pad'  # Default to use padding representation

# Set maximum number of atoms for zero padding (only valid when ATOMIC_REPR_PROCESS='pad')
MAX_ATOMS = 200  # Can be adjusted based on dataset characteristics (MAX_ATOMS = 200 covers ~95% of molecules; MAX_ATOMS = 250 covers ~97-98% of molecules)
# --- End Configuration ---

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_smiles_data(csv_path):
    """
    Load SMILES data from CSV file

    Args:
        csv_path (str): CSV file path

    Returns:
        list: List of SMILES strings
    """
    try:
        # Add keep_default_na=False to prevent 'NA' etc. from being misread
        df = pd.read_csv(csv_path, keep_default_na=False)
        if 'smiles' not in df.columns:
            logging.error(f"CSV file {csv_path} missing 'smiles' column.")
            raise ValueError("CSV file missing 'smiles' column.")
        # Handle empty strings or None
        df.dropna(subset=['smiles'], inplace=True)
        smiles_list = df['smiles'].astype(str).tolist()
        smiles_list = [s for s in smiles_list if s.strip()] # Filter empty strings
        if len(smiles_list) < len(df):
             logging.warning(f"Removed {len(df) - len(smiles_list)} empty/invalid SMILES from file {csv_path}.")
        return smiles_list
    except FileNotFoundError:
        logging.error(f"Cannot find CSV file: {csv_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}")
        raise

def extract_unimol_features(smiles_list, model_size='1.1B'):
    """
    Extract molecular representations from SMILES using Uni-Mol2

    Args:
        smiles_list (list): List of SMILES strings
        model_size (str): Model size

    Returns:
        tuple: (molecular-level representation matrix or None, atomic-level representation list or None), based on global configuration
    """
    logging.info("Initializing Uni-Mol2 model...")
    # Try to use GPU, automatically fallback to CPU if failed
    use_gpu_flag = torch.cuda.is_available()
    if not use_gpu_flag:
        logging.warning("GPU not available, will use CPU.")

    try:
        clf = UniMolRepr(
            data_type='molecule',
            remove_hs=False,
            model_name='unimolv2',
            model_size=model_size,
            use_gpu=use_gpu_flag # Dynamic setting
        )
    except Exception as e:
         logging.error(f"Failed to initialize UniMolRepr: {e}")
         # Could try forcing CPU? But more likely an environment issue
         raise # Re-raise error to let upper layer know about failure

    device = clf.device if hasattr(clf, 'device') else ('cuda' if use_gpu_flag else 'cpu')
    logging.info(f"Using device: {device}")

    mol_features = [] if EXTRACT_MOLECULAR_REPR else None
    atom_features = [] if EXTRACT_ATOMIC_REPR else None

    logging.info(f"Extracting features, molecular-level: {EXTRACT_MOLECULAR_REPR}, atomic-level: {EXTRACT_ATOMIC_REPR} (processing method: {ATOMIC_REPR_PROCESS if EXTRACT_ATOMIC_REPR else 'N/A'})...")

    feature_dim = 1536 if model_size == '1.1B' else 512 # Determine dimension based on model size

    for smiles in tqdm(smiles_list, desc="Processing SMILES"):
        try:
            # Set return_atomic_reprs parameter based on configuration
            need_atomic = EXTRACT_ATOMIC_REPR
            # Add retry logic to handle temporary network or API issues
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    repr_dict = clf.get_repr([smiles], return_atomic_reprs=need_atomic)
                    break # Success, exit retry loop
                except Exception as inner_e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Error processing SMILES {smiles} (attempt {attempt+1}/{max_retries}): {inner_e}. Retrying...")
                        import time
                        time.sleep(2) # Brief wait
                    else:
                        raise inner_e # Last attempt failed, raise exception

            # Save molecular-level representation based on configuration
            if EXTRACT_MOLECULAR_REPR:
                if 'cls_repr' in repr_dict and repr_dict['cls_repr'] is not None and len(repr_dict['cls_repr']) > 0:
                    mol_features.append(repr_dict['cls_repr'][0])
                else:
                    logging.warning(f"SMILES {smiles} did not return valid molecular-level representation, adding zero vector.")
                    mol_features.append(np.zeros(feature_dim)) # Use determined dimension

            # Save atomic-level representation based on configuration
            if EXTRACT_ATOMIC_REPR:
                if 'atomic_reprs' in repr_dict and repr_dict['atomic_reprs'] is not None and len(repr_dict['atomic_reprs']) > 0 and len(repr_dict['atomic_reprs'][0]) > 0:
                    atom_features.append(repr_dict['atomic_reprs'][0])
                else:
                    logging.warning(f"SMILES {smiles} did not return valid atomic-level representation, adding empty list.")
                    atom_features.append([]) # Add empty list to indicate failure

        except Exception as e:
            logging.error(f"Critical error processing SMILES '{smiles}', skipping this molecule: {e}")
            # Add placeholders
            if EXTRACT_MOLECULAR_REPR and mol_features is not None:
                mol_features.append(np.zeros(feature_dim)) # Use determined dimension
            if EXTRACT_ATOMIC_REPR and atom_features is not None:
                atom_features.append([])

    # Convert to numpy array if molecular-level representations were extracted
    mol_features_array = np.array(mol_features) if EXTRACT_MOLECULAR_REPR and mol_features is not None else None

    return mol_features_array, atom_features


def normalize_features(features):
    """
    Perform column-wise normalization on feature matrix (MinMaxScaler to [0, 1])

    Args:
        features (numpy.ndarray): Feature matrix (usually 2D or 3D)

    Returns:
        numpy.ndarray: Normalized feature matrix, or original matrix on error
    """
    if features is None or features.size == 0:
        logging.warning("Received empty feature matrix, cannot normalize.")
        return features

    original_shape = features.shape
    try:
        if features.ndim == 3: # e.g., (n_samples, n_atoms, n_features) atomic representations
            # Normalize each sample's atomic features independently or globally?
            # Here we choose to normalize all atomic feature values across the entire dataset together, but may need adjustment based on requirements
            logging.info("Performing global Min-Max normalization on 3D features...")
            n_samples, n_atoms, n_features = features.shape
            # Flatten for Scaler processing
            reshaped_features = features.reshape(-1, n_features)
            scaler = MinMaxScaler()
            normalized_flat = scaler.fit_transform(reshaped_features)
            # Restore shape
            normalized_features = normalized_flat.reshape(original_shape)
        elif features.ndim == 2: # e.g., (n_samples, n_features) molecular representations or pooled atomic representations
            logging.info("Performing Min-Max normalization on 2D features...")
            scaler = MinMaxScaler()
            normalized_features = scaler.fit_transform(features)
        else:
            logging.warning(f"Normalization not supported for features with dimension {features.ndim}, returning original data.")
            return features

        return normalized_features.astype(features.dtype) # Maintain original data type
    except Exception as e:
        logging.error(f"Error during feature normalization: {e}")
        return features # Return original features on error


def process_atomic_representations(atom_features, process_type='raw'):
    """
    Process atomic-level representations to convert to unified format

    Args:
        atom_features (list): List of atomic-level representations, each element is a molecule's atomic representation
        process_type (str): Processing type, options: 'raw', 'pad', 'mean_pool', 'max_pool'

    Returns:
        tuple: (processed representations, whether in npy format)
    """
    if process_type == 'raw':
        # Original representation, no processing, return pickle format
        return atom_features, False

    # Filter empty representations (these may be extraction failures)
    valid_features = [feat for feat in atom_features if isinstance(feat, np.ndarray) and feat.size > 0]

    if not valid_features:
        logging.warning("No valid atomic-level representations to process.")
        # If original list is non-empty but all invalid, need to generate placeholders
        if atom_features:
             # Try to determine dimension from configuration, otherwise use default
             feature_dim = 1536 # Assume 1.1B model
             logging.warning(f"All atomic representations are invalid, generating zero placeholders with assumed dimension {feature_dim}.")
             num_samples = len(atom_features)
             if process_type == 'pad':
                  return np.zeros((num_samples, MAX_ATOMS, feature_dim)), True
             elif process_type in ['mean_pool', 'max_pool']:
                  return np.zeros((num_samples, feature_dim)), True
             else:
                  return [], False # For raw
        else:
             return None, False # Original list is empty

    # Determine feature dimension
    feature_dim = valid_features[0].shape[-1]  # Usually 1536
    logging.info(f"Detected atomic feature dimension: {feature_dim}")

    processed_list = []
    # Use original atom_features list for iteration to maintain sample count and order
    for i, mol_feat in enumerate(atom_features):
        is_valid = isinstance(mol_feat, np.ndarray) and mol_feat.size > 0

        if not is_valid:
             # Handle invalid/empty representations, add zero placeholders
             if process_type == 'pad':
                 processed_list.append(np.zeros((MAX_ATOMS, feature_dim)))
             elif process_type in ['mean_pool', 'max_pool']:
                 processed_list.append(np.zeros(feature_dim))
             # 'raw' type already handled at function start
             continue

        # --- Process valid representations ---
        if process_type == 'pad':
            num_atoms = mol_feat.shape[0]
            if num_atoms <= MAX_ATOMS:
                padded = np.zeros((MAX_ATOMS, feature_dim), dtype=mol_feat.dtype)
                padded[:num_atoms] = mol_feat
                processed_list.append(padded)
            else:
                # Truncate
                processed_list.append(mol_feat[:MAX_ATOMS].copy()) # Use copy to avoid potential issues

        elif process_type == 'mean_pool':
            mean_feat = np.mean(mol_feat, axis=0)
            processed_list.append(mean_feat)

        elif process_type == 'max_pool':
            max_feat = np.max(mol_feat, axis=0)
            processed_list.append(max_feat)

    if not processed_list:
         logging.error("List is empty after processing atomic representations, unexpected error occurred.")
         return None, False

    try:
        # Convert to NumPy array
        processed_array = np.array(processed_list)
        logging.info(f"Atomic representation processing completed, type: {process_type}, output shape: {processed_array.shape}")
        return processed_array, True # pad, mean_pool, max_pool all return NPY compatible arrays
    except Exception as e:
         logging.error(f"Unable to convert processed atomic representation list to NumPy array: {e}")
         # If stacking fails, may need to return original list? Or decide based on situation
         # For pad mode, if still fails, returning list and False might be safer
         if process_type == 'pad':
              logging.warning("Stacking failed in padding mode, will return list and Pickle format.")
              return processed_list, False
         else: # For pooling modes, stacking failure is more unusual
              return None, False


# *** Modified process_and_save_features function signature and internal logic ***
def process_and_save_features(input_csv, output_dir, model_size='1.1B'):
    """
    Process CSV file and save features to specified output directory

    Args:
        input_csv (str): Input CSV file path
        output_dir (str): Output directory for feature files (script directory)
        model_size (str): Model size
    """
    # Get input filename (without extension) for building output filename
    base_name = os.path.splitext(os.path.basename(input_csv))[0]

    logging.info(f"Processing file: {input_csv}")

    try:
        # Load SMILES data
        smiles_list = load_smiles_data(input_csv)
        if not smiles_list:
            logging.warning(f"No valid SMILES loaded from file {input_csv}, skipping processing.")
            return

        # Extract features - get molecular-level and/or atomic-level representations based on configuration
        mol_features, atom_features = extract_unimol_features(smiles_list, model_size)

        # --- Process and save molecular-level representations ---
        if EXTRACT_MOLECULAR_REPR:
            if mol_features is None or len(mol_features) == 0:
                logging.warning(f"Failed to extract molecular-level representations for file {input_csv}.")
            else:
                processed_mol_features = mol_features.copy() # Create copy to prevent normalization from modifying original data
                if USE_NORMALIZATION:
                    logging.info("Normalizing molecular-level representations...")
                    processed_mol_features = normalize_features(processed_mol_features)

                # Build molecular-level save path (directly under output_dir)
                mol_save_path = os.path.join(output_dir, f"{base_name}_unimol_{model_size}_molecular.npy")
                try:
                    np.save(mol_save_path, processed_mol_features)
                    logging.info(f"Molecular-level representations{' (normalized)' if USE_NORMALIZATION else ''} saved to: {mol_save_path}")
                    # Verification (optional)
                    # saved_array = np.load(mol_save_path)
                    # logging.info(f"  Verification: saved shape {saved_array.shape}")
                except Exception as e:
                    logging.error(f"Error saving molecular-level features to {mol_save_path}: {e}")

        # --- Process and save atomic-level representations ---
        if EXTRACT_ATOMIC_REPR:
            if atom_features is None or len(atom_features) == 0:
                 logging.warning(f"Failed to extract atomic-level representations for file {input_csv}.")
            else:
                # Create atomic subdirectory (if it doesn't exist)
                atomic_output_dir = os.path.join(output_dir, "atomic")
                try:
                    os.makedirs(atomic_output_dir, exist_ok=True)
                except Exception as e:
                    logging.error(f"Unable to create atomic feature output directory {atomic_output_dir}: {e}. Will try to save in main output directory.")
                    atomic_output_dir = output_dir # Try fallback

                # Process atomic-level representations
                processed_atom_features, is_npy = process_atomic_representations(
                    atom_features,
                    process_type=ATOMIC_REPR_PROCESS
                )

                if processed_atom_features is None:
                     logging.warning(f"Failed to process atomic-level representations for file {input_csv}.")
                else:
                    # Normalize processed atomic representations (if needed and format compatible)
                    final_atom_features = processed_atom_features
                    if USE_NORMALIZATION and is_npy: # Only normalize npy format
                        logging.info("Normalizing processed atomic-level representations...")
                        final_atom_features = normalize_features(processed_atom_features)

                    # File suffix and processing type identifier
                    suffix = "_atomic"
                    if ATOMIC_REPR_PROCESS != 'raw':
                        suffix += f"_{ATOMIC_REPR_PROCESS}"

                    # Build atomic-level save path
                    if is_npy:
                        atom_filename = f"{base_name}_unimol_{model_size}{suffix}.npy"
                        atom_save_path = os.path.join(atomic_output_dir, atom_filename)
                        try:
                            np.save(atom_save_path, final_atom_features)
                            logging.info(f"Atomic-level representations{' (normalized)' if USE_NORMALIZATION and is_npy else ''} ({ATOMIC_REPR_PROCESS} mode) saved to: {atom_save_path}")
                            # Verification (optional)
                            # saved_array = np.load(atom_save_path)
                            # logging.info(f"  Verification: saved shape {saved_array.shape}")
                        except Exception as e:
                            logging.error(f"Error saving atomic-level features (npy) to {atom_save_path}: {e}")
                    else: # Save as pickle
                        atom_filename = f"{base_name}_unimol_{model_size}{suffix}.pkl"
                        atom_save_path = os.path.join(atomic_output_dir, atom_filename)
                        try:
                            with open(atom_save_path, 'wb') as f:
                                pickle.dump(final_atom_features, f) # Save processed (possibly raw) atomic representations
                            logging.info(f"Atomic-level representations ({ATOMIC_REPR_PROCESS} mode) saved as Pickle file: {atom_save_path}")
                        except Exception as e:
                            logging.error(f"Error saving atomic-level features (pkl) to {atom_save_path}: {e}")

    except Exception as e:
         logging.error(f"Unexpected error occurred while processing file {input_csv}: {e}")
         import traceback
         traceback.print_exc()


def main():
    """
    Main function to batch process specified CSV files for feature extraction and saving
    """
    setup_logging()

    # --- Dynamic path setup ---
    # Get the real directory of current script file
    try:
        # __file__ is available in standard Python execution
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # If __file__ is not available (e.g., in some interactive environments), fallback to current working directory, but this may be inaccurate
        logging.warning("__file__ not defined, will use current working directory as script directory. Paths may be inaccurate!")
        script_dir = os.getcwd()
    logging.info(f"Detected script directory: {script_dir}")

    # Build input data directory path (script directory up two levels -> Bilingua-DPPIV -> data)
    # Use os.path.abspath to ensure path is absolute and handle '..'
    project_base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_input_dir = os.path.join(project_base_dir, 'data')
    logging.info(f"Calculated project root directory: {project_base_dir}")
    logging.info(f"Calculated input data directory: {data_input_dir}")

    # Output directory is the script directory
    output_dir = script_dir
    logging.info(f"Output directory: {output_dir}")

    # Define target file list to process
    target_files = ['Train.csv', 'Test.csv', 'ExternalValidation.csv']

    # --- Process each target file ---
    for filename in target_files:
        input_csv_path = os.path.join(data_input_dir, filename)

        # Check if input file exists
        if not os.path.exists(input_csv_path):
            logging.warning(f"Input file {input_csv_path} does not exist, skipping this file.")
            continue

        logging.info(f"\n--- Starting to process file: {filename} ---")
        process_and_save_features(
            input_csv=input_csv_path,
            output_dir=output_dir, # Pass output directory to processing function
            model_size='1.1B'     # Or configure '512M' as needed
        )
        logging.info(f"--- Finished processing file: {filename} ---")

    logging.info("\nAll specified files processed!")

if __name__ == "__main__":
    main()