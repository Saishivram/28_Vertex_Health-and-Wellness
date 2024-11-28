import os
from flask import Flask, request, render_template, url_for
from rdkit import Chem
from rdkit.Chem import Draw

# Initialize the Flask app
app = Flask(__name__)

# Path to static folder for saving molecule images
STATIC_IMAGE_PATH = os.path.join('static', 'molecule_images')

# Dummy data for drug repurposing
drug_data = {
    "Morphine": "CN1CCC23C(C1CC2C(=O)C(=C3)O)C",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Penicillin": "CC1(C(=O)O)N(C(=O)N2CC2)C3C(C(=O)N)N(C3)C",
    # Add other drug names and SMILES here...
}

# Dummy similar drugs data
similar_drugs = [
    {"SMILES": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "Similarity": 1.0},
    {"SMILES": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "Similarity": 1.0},
    {"SMILES": "Cn1c(=O)c2nc[nH]c2n(C)c1=O", "Similarity": 0.907773},
    # Add more similar drugs with similarity scores...
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    disease_name = request.form['disease_name']
    
    # Simulating drug repurposing based on disease name
    if disease_name == "covid":
        selected_drug_name = "Caffeine"
        selected_drug_smiles = drug_data["Caffeine"]
    else:
        selected_drug_name = "Morphine"
        selected_drug_smiles = drug_data["Morphine"]

    # Generate molecule image
    molecule = Chem.MolFromSmiles(selected_drug_smiles)
    image_path = os.path.join(STATIC_IMAGE_PATH, 'generated_molecule.png')
    Draw.MolToFile(molecule, image_path)
    
    # Find similar drugs and compute similarity percentage
    similar_drug_results = []
    for drug in similar_drugs:
        similarity_percentage = round(drug["Similarity"] * 100, 2)
        similar_drug_results.append({
            "SMILES": drug["SMILES"],
            "SimilarityPercentage": similarity_percentage
        })

    # Return the results page with the necessary data
    return render_template(
        'results.html',
        disease_name=disease_name,
        selected_drug_name=selected_drug_name,
        selected_drug_smiles=selected_drug_smiles,
        molecule_image=url_for('static', filename='molecule_images/generated_molecule.png'),
        similar_drug_results=similar_drug_results
    )

if __name__ == '_main_':
    # Ensure static image directory exists
    os.makedirs(STATIC_IMAGE_PATH, exist_ok=True)
    app.run(debug=True)