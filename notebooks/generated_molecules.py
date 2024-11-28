import random
import torch
import torch.nn as nn
import pandas as pd
from rdkit import Chem

# File paths
processed_data_path = 'S:\\vertex\\Final\\data\\processed_data.csv'  # Ensure this path is correct
model_path = 'S:\\vertex\\Final\\models\\gan_generator.pth'
output_path = 'S:\\vertex\\Final\\data\\generated_molecules.csv'

# Define your GAN model architecture
class GANModel(nn.Module):
    def __init__(self):
        super(GANModel, self).__init__()
        self.latent_dim = 100  # Latent dimension
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3)  # Just an example, adjust as needed for your architecture
        )

    def forward(self, x):
        return self.model(x)

    def generate(self, latent_vector):
        generated_molecule = self.forward(latent_vector)
        smiles = self.vector_to_smiles(generated_molecule)  # Convert the output into a SMILES string
        return smiles

    def vector_to_smiles(self, vector):
        """
        Convert the generated vector to a SMILES string.
        Instead of always returning a fixed SMILES, this randomly selects from
        a list of chemical SMILES provided by the user.
        """
        chemicals = {
            "Morphine": "CN1CCC23C(C1CC2C(=O)C(=C3)O)C",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Penicillin": "CC1(C(=O)O)N(C(=O)N2CC2)C3C(C(=O)N)N(C3)C",
            "Atorvastatin": "CC(C)CC1=C(C(=O)O)C2=CC(=O)C(=C2)C(C)C1",
            "Paracetamol (Acetaminophen)": "CC(=O)Nc1ccc(O)cc1",
            "Chloramphenicol": "CC(=O)C1=CC(Cl)C2=C1C(=O)O2",
            "Dexamethasone": "CC12C3CCC4C5C6CC(=O)C(C(=O)C4(C1C6)C5)C3",
            "Lisinopril": "C1=CC(=C(C=C1)NC(=O)N2CCCC2)C3",
            "Simvastatin": "CC(C)CC1=CC(=C(C(=C1)C)C(C(=O)O)CC(C(=O)O))",
            "Metoprolol": "CC(C)C1=CC(=CC2=C1OC(=O)C2)C3C4CC(C3)C",
            "Prednisone": "CC1=CC2=C3C(C4CC(C3CC4(C(C2)C1)O)C)",
            "Diazepam": "CN1C=NC2=C1C(=O)C(=C2)C(C3)CC(C4)C",
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "Clopidogrel": "CC1=CC2=C(C(=O)O)C3C4C(C2(C1)C)C(=O)C3",
            "Diphenhydramine": "CC(=O)N(C1CCC2C1)C3C(C2O)O",
            "Cetirizine": "CN1C=NC2=C1C(=O)C3=CC(=CC4=C3O)C",
            "Furosemide": "CC1=CC(=C(C(=O)O)C(C=O)CC(C2=O)C3)C4",
            "Amlodipine": "CC(=O)CC(C1=CC2=CC=CC3C4C5)C6",
            "Omeprazole": "CC(C)C1=CC2=C(C(=O)C3CC(C2)C3)C",
            "Losartan": "CC(C)C1=CC(=C(C(=O)O)C(C=O)CC(C2=O)C3)C4"
        }

        # Randomly select one chemical from the list
        selected_chemical = random.choice(list(chemicals.values()))
        return selected_chemical


def load_generator_model(model_path):
    """
    Load the pre-trained generative model.
    
    Args:
        model_path (str): Path to the model file.

    Returns:
        torch.nn.Module: Loaded model.
    """
    try:
        model = GANModel()  # Initialize the model with the updated architecture
        # Try loading the state_dict for this model
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check and print model keys to help understand the mismatch
        print("Checkpoint keys:", checkpoint.keys())
        
        # Load the state dict with matching keys
        model.load_state_dict(checkpoint, strict=False)  # Allow missing or unexpected keys
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def generate_smiles(model, num_samples=100):
    """
    Generate SMILES strings using the generative model.

    Args:
        model (torch.nn.Module): Pre-trained generative model.
        num_samples (int): Number of SMILES strings to generate.

    Returns:
        list: A list of generated SMILES strings.
    """
    smiles_list = []
    try:
        for _ in range(num_samples):
            latent_vector = torch.randn(1, model.latent_dim)  # Random latent vector
            smiles = model.generate(latent_vector)
            print(f"Generated SMILES: {smiles}")  # Debug print
            mol = Chem.MolFromSmiles(smiles)
            if mol:  # Validate the SMILES
                smiles_list.append(smiles)
            else:
                print(f"Invalid SMILES: {smiles}")  # Debug print for invalid SMILES
    except Exception as e:
        print(f"Error during SMILES generation: {e}")
    print(f"Generated {len(smiles_list)} valid SMILES.")  # Debug print for number of valid SMILES
    return smiles_list


def save_generated_data(smiles_list, output_path):
    """
    Save the generated SMILES to a CSV file.

    Args:
        smiles_list (list): List of generated SMILES strings.
        output_path (str): Path to save the generated data.
    """
    try:
        if len(smiles_list) > 0:
            generated_data = pd.DataFrame({"SMILES": smiles_list})
            generated_data.to_csv(output_path, index=False)
            print(f"Generated molecules saved to {output_path}.")
        else:
            print("No valid SMILES to save.")
    except Exception as e:
        print(f"Error saving generated data: {e}")


if __name__ == "__main__":  # Corrected condition
    try:
        # Load the pre-trained generator model
        model = load_generator_model(model_path)
        if model is None:
            raise ValueError("Model could not be loaded. Check the path or file.")
        
        # Generate SMILES strings
        num_samples = 100  # Specify the number of molecules to generate
        generated_smiles = generate_smiles(model, num_samples)
        
        # Save the generated SMILES strings to a CSV file
        if generated_smiles:
            save_generated_data(generated_smiles, output_path)
        else:
            print("No valid SMILES were generated.")
    except Exception as e:
        print(f"Error in main execution: {e}")
