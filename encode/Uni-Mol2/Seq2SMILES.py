from rdkit import Chem
import pandas as pd
import os

def build_peptide_smiles(sequence):
    """
    Build the SMILES string for a peptide sequence using RDKit's FASTA parser.
    
    Args:
        sequence (str): Amino acid sequence using single-letter codes.
        
    Returns:
        str: SMILES representation of the peptide, or None if conversion fails.
    """
    try:
        # Convert sequence to RDKit molecule using FASTA parser
        mol = Chem.MolFromFASTA(sequence)
        if mol is None:
            return None
        
        # Convert to SMILES
        return Chem.MolToSmiles(mol)
    
    except Exception as e:
        print(f"Error converting sequence {sequence} to SMILES: {str(e)}")
        return None

def convert_csv_to_smiles(input_file, output_dir):
    """
    Convert peptide sequences in a CSV file to SMILES and save to a new CSV file.
    
    Args:
        input_file (str): Path to input CSV file containing sequences
        output_dir (str): Directory to save the output CSV file
    """
    try:
        # Read input CSV file, preventing 'NA' from being interpreted as NaN
        df = pd.read_csv(input_file, keep_default_na=False)
        original_count = len(df)
        
        # Convert sequences to SMILES
        df['smiles'] = df['seq'].apply(build_peptide_smiles)
        
        # Identify failed conversions
        failed_conversions = df[df['smiles'].isnull()]
        num_failed = len(failed_conversions)
        
        if num_failed > 0:
            print(f"Warning: {num_failed} sequences failed conversion in {input_file}:")
            for index, row in failed_conversions.iterrows():
                print(f"  - Sequence: {row['seq']}")
        
        # Drop rows where SMILES conversion failed
        df_clean = df.dropna(subset=['smiles'])
        final_count = len(df_clean)
        
        # Select and reorder columns
        df_clean = df_clean[['smiles', 'label']]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file path with same name as input file
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        
        # Save to CSV
        df_clean.to_csv(output_file, index=False)
        
        print(f"Successfully processed {input_file}")
        print(f"Original sequences: {original_count}")
        print(f"Converted {final_count} sequences to SMILES")
        if num_failed > 0:
            print(f"Dropped {num_failed} sequences due to conversion errors.")
        print(f"Output saved to {output_file}\n")
        
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}\n")

def process_all_csv_files(input_dir, output_dir):
    """
    Process all CSV files in the input directory and convert sequences to SMILES.
    
    Args:
        input_dir (str): Directory containing input CSV files
        output_dir (str): Directory to save output CSV files
    """
    # Get all CSV files in input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        convert_csv_to_smiles(input_path, output_dir)

if __name__ == "__main__":
    # Set input and output directories (using relative paths)
    input_dir = "../../data"  # Relative path to data directory
    output_dir = "./SMILES_output"  # Output to current directory
    
    # Process all CSV files
    process_all_csv_files(input_dir, output_dir)