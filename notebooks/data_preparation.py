import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# File paths
raw_data_path = 'S:\\vertex\\Final\\data\\raw_data.csv'
processed_data_path = 'S:\\vertex\\Final\\data\\processed_data.csv'

def validate_and_process_smiles(smiles):
    """
    Validate SMILES strings and compute properties for valid molecules.

    Args:
        smiles (str): SMILES string to validate and process.

    Returns:
        dict: A dictionary with molecular properties or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Compute properties
        mol_weight = Descriptors.MolWt(mol)
        log_p = Descriptors.MolLogP(mol)

        return {
            "SMILES": smiles,
            "MolecularWeight": mol_weight,
            "LogP": log_p
        }
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

def process_data(input_path, output_path):
    """
    Process the raw data, validate SMILES, and compute molecular properties.
    Also includes the target column for training.

    Args:
        input_path (str): Path to the raw dataset.
        output_path (str): Path to save the processed dataset.

    Returns:
        pd.DataFrame: Processed dataset with valid molecules and their properties.
    """
    try:
        # Load raw data
        raw_data = pd.read_csv(input_path)
        print("Columns in raw data:", raw_data.columns)  # Debugging line to check columns
        
        if 'canonical_smiles' not in raw_data.columns:
            raise ValueError("The input dataset must contain a 'canonical_smiles' column.")
        
        if 'pchembl_value' not in raw_data.columns:
            raise ValueError("The input dataset must contain a 'pchembl_value' column as the target.")
        
        # Check for missing values in important columns
        print("Missing values in canonical_smiles and pchembl_value:", raw_data[['canonical_smiles', 'pchembl_value']].isnull().sum())
        
        processed_records = []
        for _, row in raw_data.iterrows():
            smiles = row['canonical_smiles']
            activity = row.get('pchembl_value', None)  # Using 'pchembl_value' as the target column
            
            # Debug print
            print(f"Processing SMILES: {smiles}")

            if activity is not None:
                record = validate_and_process_smiles(smiles)
                if record:
                    record['Activity'] = activity  # Add the activity value to the record
                    processed_records.append(record)

        # Check if any records were processed
        if not processed_records:
            raise ValueError("No valid records were processed.")
        
        # Convert to DataFrame
        processed_data = pd.DataFrame(processed_records)

        # Check if all required columns are present
        if not {'MolecularWeight', 'LogP', 'Activity'}.issubset(processed_data.columns):
            raise ValueError("Processed data is missing required columns: 'MolecularWeight', 'LogP', 'Activity'.")
        
        # Log the number of records before saving
        print(f"Processed {len(processed_data)} records.")

        # Save processed data
        processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}.")
        return processed_data

    except Exception as e:
        print(f"Error in data processing: {e}")
        return pd.DataFrame()

# Corrected main execution check
if __name__ == "__main__":  # Corrected line here
    try:
        # Process the raw data
        processed_data = process_data(raw_data_path, processed_data_path)
        
        if not processed_data.empty:
            print(f"Successfully processed {len(processed_data)} molecules.")
        else:
            print("No valid molecules found in the dataset.")
    except Exception as e:
        print(f"Error in main execution: {e}")
