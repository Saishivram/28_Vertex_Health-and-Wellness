from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import Draw
import pandas as pd
import os

# File paths
generated_data_path = 'S:\\vertex\\Final\\data\\generated_molecules.csv'
processed_data_path = 'S:\\vertex\\Final\\data\\processed_data.csv'
image_output_path = 'S:\\vertex\\Final\\app\\static\\molecule_images\\molecule_image.png'
similar_drugs_output_path = 'S:\\vertex\\Final\\data\\similar_drugs.csv'

# Ensure the output directory exists
os.makedirs(image_output_path, exist_ok=True)

# Load data
try:
    generated_data = pd.read_csv(generated_data_path)
    processed_data = pd.read_csv(processed_data_path)

    if 'SMILES' not in generated_data.columns:
        raise ValueError(f"'SMILES' column not found in {generated_data_path}.")
    if 'SMILES' not in processed_data.columns:
        raise ValueError(f"'SMILES' column not found in {processed_data_path}.")
except Exception as e:
    print(f"Error loading data files: {e}")
    exit(1)

def compute_tanimoto_similarity(smiles1, smiles2):
    """
    Compute Tanimoto similarity between two SMILES strings.
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if not mol1 or not mol2:
            raise ValueError(f"Invalid SMILES: {smiles1} or {smiles2}")
        
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)
        return DataStructs.FingerprintSimilarity(fp1, fp2)
    except Exception as e:
        print(f"Error computing Tanimoto similarity: {e}")
        return None

def save_molecule_image(smiles, filename):
    """
    Generate and save an image of a molecule from its SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            Draw.MolToFile(mol, filename)
        else:
            print(f"Invalid SMILES for image generation: {smiles}")
    except Exception as e:
        print(f"Error generating image: {e}")

def find_similar_drugs(target_smiles, threshold=0.7, save_images=False):
    """
    Find drugs in the processed dataset similar to the target molecule.
    
    Args:
        target_smiles (str): SMILES string of the target molecule.
        threshold (float): Minimum Tanimoto similarity for a drug to be considered similar.
        save_images (bool): Whether to save images of similar drugs.

    Returns:
        pd.DataFrame: DataFrame of similar drugs with their SMILES and similarity scores.
    """
    try:
        target_mol = Chem.MolFromSmiles(target_smiles)
        if not target_mol:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")

        similar_drugs = []

        for _, row in processed_data.iterrows():
            smiles = row.get('SMILES')
            if not smiles:
                continue
            
            similarity = compute_tanimoto_similarity(target_smiles, smiles)
            if similarity and similarity >= threshold:
                similar_drugs.append({'SMILES': smiles, 'Similarity': similarity})
                
                if save_images:
                    # Save the molecule image
                    sanitized_smiles = smiles.replace('/', '_')
                    image_filename = os.path.join(image_output_path, f"{sanitized_smiles}.png")
                    save_molecule_image(smiles, image_filename)

        return pd.DataFrame(similar_drugs).sort_values(by='Similarity', ascending=False)

    except Exception as e:
        print(f"Error in finding similar drugs: {e}")
        return pd.DataFrame()

# Example Usage
if __name__ == "__main__":
    try:
        # Assume target molecule is the first molecule in the generated dataset
        if generated_data.empty:
            raise ValueError("Generated molecules dataset is empty.")
        
        target_smiles = generated_data['SMILES'].iloc[0]
        threshold = 0.7  # Set the similarity threshold

        similar_drugs_df = find_similar_drugs(target_smiles, threshold, save_images=True)

        # Save the similar drugs to a CSV file
        similar_drugs_df.to_csv(similar_drugs_output_path, index=False)

        print(f"Similar drugs saved to {similar_drugs_output_path}.")
        print(f"Images saved in {image_output_path}.")
    except Exception as e:
        print(f"Error in main execution: {e}")