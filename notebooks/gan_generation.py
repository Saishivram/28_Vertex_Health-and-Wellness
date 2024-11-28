import torch
import pandas as pd
from rdkit import Chem
from gan_training import Generator  # Ensure this imports the correct Generator model.

# Load the trained generator model
generator = Generator()
generator.load_state_dict(torch.load('S:\\vertex\\Final\\models\\gan_generator.pth'))
generator.eval()

# Generate SMILES strings
def generate_smiles(model, num_samples=100):
    smiles_list = []
    for _ in range(num_samples):
        z = torch.randn(1, 100)  # Latent vector (random noise)
        fake_molecule = model(z)  # Generate fake molecular features
        fake_molecule = fake_molecule.detach().numpy()  # Convert tensor to numpy array

        # Assuming the fake_molecule output is a 3D feature vector, convert it to a SMILES string somehow
        # Here, we need to define the conversion from features (e.g., MW, LogP) to SMILES.
        # In this placeholder, I use a mock function `features_to_smiles()`, but you must implement this.
        smiles = features_to_smiles(fake_molecule)  # This should be a custom conversion function

        # Validate and store if valid (use RDKit to check if SMILES is valid)
        if validate_smiles(smiles):
            smiles_list.append(smiles)
    return smiles_list

# Mock function to convert molecular features to SMILES
# You need to replace this with a real function depending on how your model works.
def features_to_smiles(features):
    # This is a placeholder for conversion logic.
    # You could try a model that converts molecular features back into SMILES, such as an inverse SMILES model
    return "C1=CC=CC=C1"  # Example SMILES for Benzene

# Validate SMILES strings
def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Save generated SMILES to CSV
def save_generated_molecules(smiles_list, output_path):
    df = pd.DataFrame(smiles_list, columns=["SMILES"])
    df.to_csv(output_path, index=False)
    print(f"Generated molecules saved to {output_path}")

# Generate and save molecules
num_samples = 150  # Generate 150 molecules
generated_smiles = generate_smiles(generator, num_samples)
valid_smiles = [smiles for smiles in generated_smiles if validate_smiles(smiles)]
save_generated_molecules(valid_smiles, 'S:\\vertex\\Final\\data\\generated_molecules.csv')
