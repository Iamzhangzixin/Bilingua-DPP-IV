import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig, ESMProteinError, LogitsOutput
from esm.sdk import batch_executor
import os
from tqdm import tqdm
import time
import traceback # For printing detailed error information

# Set whether to use GPU, if GPU is not available, use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU for computation.")

# Set whether to perform normalization
# USE_NORMALIZATION = True  # Set to True for column-wise normalization, False for no normalization
USE_NORMALIZATION = False

# # Set API token (passed from main function to make_vector_esm_6b)
# API_TOKEN = "YOUR_API_TOKEN_HERE"  # No longer defined as global variable here

def pad_or_truncate_sequence(seq, max_length=40):
    """
    Pad or truncate sequences to ensure uniform length of max_length.
    Note: ESM3 API automatically handles padding, but truncation still needs to be done manually.

    Args:
        seq (str): Original peptide sequence.
        max_length (int): Target maximum sequence length.

    Returns:
        str: Processed sequence (if truncation is needed).
    """
    if not isinstance(seq, str): # Add type checking
        print(f"Warning: Received non-string sequence: {seq} (type: {type(seq)}), will attempt conversion.")
        try:
            seq = str(seq)
        except:
             print(f"Error: Cannot convert sequence {seq} to string. Skipping this sequence.")
             return None # Return None to indicate processing failure

    if len(seq) > max_length:
        # Truncate to specified length
        return seq[:max_length]
    else:
        # If less than or equal to max_length, return original sequence, let API handle padding
        return seq

def embed_sequence(client: ESM3ForgeInferenceClient, sequence: str) -> LogitsOutput | ESMProteinError:
    """
    Extract feature vectors for a single sequence using Forge API (for 6B model).
    Modified return type to allow error propagation.

    Args:
        client: ESM3ForgeInferenceClient instance
        sequence: Protein sequence string

    Returns:
        LogitsOutput | ESMProteinError: Output object containing embedding vectors or error object
    """
    try:
        protein = ESMProtein(sequence=sequence)
        # Note: client.encode now directly returns tensor or error
        protein_tensor = client.encode(protein)
        if isinstance(protein_tensor, ESMProteinError):
            # print(f"Debug: encode error for seq {sequence}: {protein_tensor}")
            return protein_tensor # Return error directly

        output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        # print(f"Debug: Logits successful for {sequence}")
        return output
    except Exception as e:
         # Catch any other unexpected errors that may occur within embed_sequence
         print(f"Caught unexpected error in embed_sequence({sequence}): {e}")
         # Return a generic error object or marker so upstream knows about failure
         # Here we construct an ESMProteinError to maintain type consistency (though reason may differ)
         return ESMProteinError(message=f"Unexpected error in embed_sequence: {e}", protein=ESMProtein(sequence=sequence))


def make_vector_esm_6b(sequences, batch_size=20, api_token="YOUR_DEFAULT_TOKEN", max_seq_len_for_padding=40):
    """
    Batch extract 3D feature vectors from protein sequences using ESMC 6B model.
    Fixed the issue of different sequences generating embeddings with different shapes.
    Added error handling and more detailed logging.

    Args:
        sequences (list): List of processed protein sequences.
        batch_size (int): Batch size for controlling API request frequency.
        api_token (str): Token for API authentication.
        max_seq_len_for_padding (int): Maximum length for final numpy array padding.

    Returns:
        np.ndarray | None: Feature vector matrix, or None on complete failure.
    """
    valid_sequences = [s for s in sequences if s is not None] # Filter out None returned by pad_or_truncate
    if not valid_sequences:
        print("Error: No valid sequences to process.")
        return None

    num_original_sequences = len(sequences)
    num_valid_sequences = len(valid_sequences)
    if num_valid_sequences < num_original_sequences:
        print(f"Warning: {num_original_sequences - num_valid_sequences} sequences removed due to processing failure.")

    print(f"Initializing ESMC 6B client...")
    try:
        client = ESM3ForgeInferenceClient(
            model="esmc-6b-2024-12",
            url="https://forge.evolutionaryscale.ai",
            token=api_token
        )
    except Exception as e:
        print(f"Error: Failed to initialize ESM3ForgeInferenceClient: {e}")
        return None

    all_embeddings = [] # Store (original_index, embedding_vector)
    embedding_shapes = {}
    observed_max_seq_len = 0 # Record maximum length actually returned from API
    processed_indices = set() # Record successfully processed original sequence indices

    # Process sequences in batches to avoid API limits
    original_indices = [i for i, s in enumerate(sequences) if s is not None] # Get original indices of valid sequences

    for i in range(0, num_valid_sequences, batch_size):
        batch_indices = original_indices[i:i+batch_size]
        batch_sequences = [valid_sequences[k] for k in range(i, min(i+batch_size, num_valid_sequences))] # Get valid sequences for current batch
        print(f"Processing batch {i//batch_size + 1}/{(num_valid_sequences-1)//batch_size + 1}, {len(batch_sequences)} sequences (original indices {batch_indices[0]} to {batch_indices[-1]})")

        batch_outputs = None # Initialize as None
        try:
            with batch_executor() as executor:
                # execute_batch now directly returns result list (may contain LogitsOutput or ESMProteinError)
                batch_outputs = executor.execute_batch(
                    user_func=embed_sequence,
                    client=client,
                    sequence=batch_sequences
                )
        except Exception as e:
            print(f"Error in batch executor: {e}")
            print(f"Skipping batch {i//batch_size + 1}")
            # Optionally record failed indices
            continue # Process next batch

        if batch_outputs is None:
             print(f"Warning: Batch {i//batch_size + 1} returned no output.")
             continue

        # Process batch results
        successful_in_batch = 0
        for j, output in enumerate(batch_outputs):
            original_index = batch_indices[j] # Get current sequence's index in original list

            if isinstance(output, LogitsOutput) and hasattr(output, 'embeddings') and output.embeddings is not None:
                embedding = output.embeddings.cpu().numpy() # (1, seq_len, dim)

                # Ensure it's 3D
                if len(embedding.shape) != 3 or embedding.shape[0] != 1:
                     print(f"Warning: Embedding shape {embedding.shape} for sequence index {original_index} ('{batch_sequences[j]}') does not match expected (1, seq_len, dim). Skipping this embedding.")
                     continue

                seq_len = embedding.shape[1]
                shape_key = str(embedding.shape)
                embedding_shapes[shape_key] = embedding_shapes.get(shape_key, 0) + 1

                if seq_len > observed_max_seq_len:
                    observed_max_seq_len = seq_len

                all_embeddings.append((original_index, embedding[0])) # Store (original_index, (seq_len, dim) numpy array)
                processed_indices.add(original_index)
                successful_in_batch += 1
            elif isinstance(output, ESMProteinError):
                 print(f"Warning: Failed to get embedding for sequence index {original_index} ('{output.protein.sequence if output.protein else batch_sequences[j]}'): {output.message}")
            else:
                 # Handle other possible failure cases (e.g., embed_sequence returns None or other types)
                 print(f"Warning: Sequence index {original_index} ('{batch_sequences[j]}') returned unexpected output type: {type(output)}.")

        print(f"Batch processing completed, successfully obtained {successful_in_batch} / {len(batch_sequences)} embeddings.")
        # (Optional) Add brief sleep just in case, but batch_executor usually handles rate limiting
        # time.sleep(0.5)

    if not all_embeddings:
        print("Error: No valid embedding vectors obtained after processing all sequences.")
        return None

    print("\n--- Embedding Post-processing ---")
    print("Embedding shape statistics:")
    for shape, count in embedding_shapes.items():
        print(f"  {shape}: {count} sequences")
    print(f"Maximum sequence length observed (from API): {observed_max_seq_len}")

    # Determine final length for padding
    final_pad_len = max(max_seq_len_for_padding, observed_max_seq_len)
    print(f"Final sequence length for padding: {final_pad_len}")

    # Sort embeddings by original index and prepare for padding
    all_embeddings.sort(key=lambda x: x[0])
    padded_embeddings_map = {} # Use dictionary for storage, key is original index
    final_embedding_dim = -1

    for original_index, emb_2d in all_embeddings:
        seq_len, dim = emb_2d.shape

        if final_embedding_dim == -1:
            final_embedding_dim = dim
            print(f"Recorded embedding dimension as: {final_embedding_dim}")
        elif dim != final_embedding_dim:
            print(f"Warning: Embedding dimension ({dim}) for index {original_index} does not match expected dimension ({final_embedding_dim}). Skipping this embedding.")
            processed_indices.discard(original_index) # Remove from successful set
            continue

        # Create padded embedding
        padded_emb = np.zeros((final_pad_len, final_embedding_dim), dtype=emb_2d.dtype)
        current_len = min(seq_len, final_pad_len) # Ensure not exceeding padding length
        padded_emb[:current_len, :] = emb_2d[:current_len, :]
        padded_embeddings_map[original_index] = padded_emb

    # Build final feature matrix, maintaining order corresponding to original sequence list
    # For sequences that failed processing, fill with zero vectors
    final_feature_list = []
    num_failures = 0
    if final_embedding_dim == -1 and not padded_embeddings_map:
         print("Error: No valid embeddings to determine dimension, cannot create zero vectors.")
         return None
    elif final_embedding_dim == -1 and padded_embeddings_map:
         # If final_embedding_dim was never set but there are embeddings, this is a logic error
         print("Internal error: Embeddings exist but final_embedding_dim not set.")
         return None

    zero_vector = np.zeros((final_pad_len, final_embedding_dim), dtype=np.float32) # Assume float32

    for i in range(num_original_sequences):
        if i in processed_indices and i in padded_embeddings_map:
            final_feature_list.append(padded_embeddings_map[i])
        else:
            # If original sequence was valid but processing failed, or original sequence was invalid
            final_feature_list.append(zero_vector)
            if sequences[i] is not None: # If it's a valid sequence that failed processing
                num_failures += 1

    if num_failures > 0:
         print(f"Warning: {num_failures} valid sequences failed during processing or padding phase, replaced with zero vectors.")
    print(f"Final number of sequences successfully processed and included in feature matrix: {len(processed_indices)}")

    if not final_feature_list:
         print("Error: Unable to build final feature list.")
         return None

    try:
        seq_feats = np.stack(final_feature_list, axis=0)
        print(f"Final feature matrix shape: {seq_feats.shape}")
        return seq_feats
    except ValueError as e_stack:
        print(f"Error: Unable to stack final embedding vectors, possibly inconsistent shapes: {e_stack}")
        # Further debug information
        #for idx, arr in enumerate(final_feature_list):
        #    print(f"  Feature {idx} shape: {arr.shape}, dtype: {arr.dtype}")
        return None
    except Exception as e_final:
         print(f"Error: Unknown error occurred while building final feature matrix: {e_final}")
         return None


def normalize_features(features):
    """
    Perform min-max normalization on feature matrix by columns.
    Warning: This method may not be standard practice for 3D features.

    Args:
        features (np.ndarray): Input feature matrix with shape (num_samples, sequence_length, feature_dim).

    Returns:
        np.ndarray: Normalized feature matrix.
    """
    if features.ndim != 3:
        print("Error: normalize_features requires 3D numpy array.")
        return features # Return as is

    original_shape = features.shape
    num_samples = features.shape[0]
    # Flatten each sample's (seq_len, dim) into a long vector for normalization
    # Note: This normalizes all feature values across the entire dataset together, not each sample independently
    reshaped_for_scaling = features.reshape(num_samples, -1)
    scaler = MinMaxScaler()
    normalized_flat = scaler.fit_transform(reshaped_for_scaling)
    # Reshape back to original 3D shape
    normalized = normalized_flat.reshape(original_shape)
    return normalized

def main():
    """
    Main function to batch process Train.csv, Test.csv, ExternalValidation.csv for feature extraction and saving.
    """
    # --- Configuration ---
    # Set base data path (use relative path)
    base_data_dir = "../../data"  # Relative path to data directory
    # Define target file list
    target_files = ['Train.csv', 'Test.csv', 'ExternalValidation.csv']
    # Set whether to perform normalization (False means no normalization)
    USE_NORMALIZATION = False
    # Set API token (please ensure this is valid)
    API_TOKEN = "YOUR_API_TOKEN_HERE"  # Replace with your actual API token
    # Set model name (currently only processing 6b)
    model_name = "6b"
    # Maximum sequence length (for truncation and final padding)
    max_seq_len = 40
    # Batch size
    batch_size = 20 # Can be adjusted based on API limits and memory
    # --- End Configuration ---

    print("--- Starting batch processing for ESMC 6B feature extraction ---")

    # Determine output base directory (based on normalization setting)
    if USE_NORMALIZATION:
        output_base_dir = os.path.join(".", "ESMC_features", "normalized")
    else:
        output_base_dir = os.path.join(".", "ESMC_features", "raw")

    # Create output base directory (if it doesn't exist)
    try:
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"Ensuring output base directory exists: {output_base_dir}")
    except Exception as e:
        print(f"Error: Unable to create output base directory {output_base_dir}: {e}")
        return # Exit if unable to create directory

    # --- Process each target file ---
    for filename in target_files:
        input_csv = os.path.join(base_data_dir, filename)
        print(f"\n=========================================")
        print(f"=== Processing file: {filename} ===")
        print(f"=========================================")
        print(f"Input file path: {input_csv}")

        # Check if input file exists
        if not os.path.exists(input_csv):
            print(f"Warning: Input file {input_csv} does not exist, skipping this file.")
            continue

        # --- File processing workflow ---
        try:
            # Read data and prevent pandas from treating 'NA' as NaN
            try:
                data = pd.read_csv(input_csv, keep_default_na=False)
                print(f"Successfully read data, {len(data)} records total. (keep_default_na=False)")
            except Exception as e:
                print(f"Error reading CSV file {filename}: {e}")
                continue # Skip to next file

            if 'seq' not in data.columns:
                print(f"CSV file {filename} missing 'seq' column. Skipping this file.")
                continue

            # Check for empty sequences or unexpected types
            # Note: Since keep_default_na=False, we need to check for actual empty strings '' or None (if present in source file)
            original_count = len(data)
            # Explicitly handle empty strings and None (if needed)
            data = data[data['seq'].notna()] # Remove Python None
            data = data[data['seq'].astype(str).str.strip() != ''] # Remove empty strings or strings with only spaces
            if len(data) < original_count:
                 print(f"Warning: Removed {original_count - len(data)} rows because 'seq' column was empty or contained only spaces.")

            sequences = data['seq'].tolist() # Use tolist() to get list
            print(f"Total valid sequences: {len(sequences)}")

            if not sequences:
                 print("No valid sequences in file. Skipping this file.")
                 continue

            # Pad or truncate sequences
            processed_sequences = [pad_or_truncate_sequence(seq, max_length=max_seq_len) for seq in sequences]
            # Filter out sequences that failed processing (cases where pad_or_truncate_sequence returns None)
            num_processed_before_filter = len(processed_sequences)
            processed_sequences_filtered = [s for s in processed_sequences if s is not None]
            num_processed_after_filter = len(processed_sequences_filtered)
            if num_processed_after_filter < num_processed_before_filter:
                 print(f"Warning: Removed {num_processed_before_filter - num_processed_after_filter} sequences during processing (truncation/conversion) phase.")

            print(f"Sequences preprocessed (truncated to {max_seq_len}). Valid sequences after processing: {num_processed_after_filter}")
            if not processed_sequences_filtered:
                 print("No valid sequences after preprocessing. Skipping this file.")
                 continue
            # print(f"Sample processed sequences: {processed_sequences_filtered[:3]}...") # Print few samples

            # Extract features (only processing 6b model)
            print(f"\nExtracting features using ESM {model_name}...")
            esm_matrix = make_vector_esm_6b(
                processed_sequences_filtered, # Use filtered valid sequence list
                batch_size=batch_size,
                api_token=API_TOKEN,
                max_seq_len_for_padding=max_seq_len # Pass length for final padding
            )

            if esm_matrix is None:
                 print(f"Error: Failed to generate feature matrix for file {filename}. Skipping save.")
                 continue # Skip to next file

            print(f"\nFeature extraction completed.")
            print(f"Feature matrix shape: {esm_matrix.shape}")
            print(f"Feature matrix dtype: {esm_matrix.dtype}")

            # If normalization is needed (currently configured as False)
            if USE_NORMALIZATION:
                print("Normalizing features by column...")
                esm_matrix = normalize_features(esm_matrix)
                if esm_matrix is not None: # Ensure normalization didn't fail
                    print("Normalization completed.")
                else:
                    print("Error: Normalization failed. Skipping save.")
                    continue

            # Build save path
            base_name = os.path.splitext(filename)[0] # Get base name from current filename
            save_path = os.path.join(output_base_dir, f"{base_name}_esm_{model_name}.npy")

            # Save features
            try:
                np.save(save_path, esm_matrix)
                print(f"Saved {'normalized' if USE_NORMALIZATION else 'raw'} features to: {save_path}")
            except Exception as e_save:
                print(f"Error: Failed to save feature file {save_path}: {e_save}")
                continue # Also skip to next if save fails

            # (Optional) Verify saved array
            try:
                saved_array = np.load(save_path)
                print(f"Verification: Saved array shape: {saved_array.shape}")
                # print(f"Verification: Saved array dtype: {saved_array.dtype}")
                # print("Verification: Saved array sample:")
                # print(saved_array[:1, :3, :4]) # Print fewer samples
                if saved_array.shape != esm_matrix.shape:
                     print(f"Warning: Loaded array shape {saved_array.shape} doesn't match in-memory array shape {esm_matrix.shape}!")
            except Exception as e_verify:
                 print(f"Warning: Error verifying saved file {save_path}: {e_verify}")

        except Exception as e_main_loop:
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Critical error occurred while processing file {filename}: {str(e_main_loop)}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            traceback.print_exc() # Print full error stack
            continue # Continue trying to process next file

    print("\n--- Batch processing completed ---")

if __name__ == "__main__":
    main()