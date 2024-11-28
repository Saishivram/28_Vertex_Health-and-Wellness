# Problem Statement:
Millions of people worldwide suffer from rare diseases, facing limited treatment options, high costs, and slow drug development.

AI-driven drug discovery can accelerate treatment development, reduce costs, and make therapies more accessible for underserved populations.

                
                        Input Page(Accepting the disease)
![image](https://github.com/user-attachments/assets/545b0aad-4b1a-4835-be59-d85199eca31b)

Trained AI Model Based on the dataset and displayed the following results

                        Output Page(Containing all information)
![image](https://github.com/user-attachments/assets/4488b6a1-cb5f-4795-882e-13f72505ad45)





Drug Discovery Powered by 
AI
 Unlocking Hidden Cures: AI at the Frontier of Drug Discovery and Repurposing.
Problem Statement
 Millions of individuals with rare diseases suffer from a lack of effective treatments, 
exacerbated by the high costs and slow progress of traditional drug discovery methods. 
Our goal is to leverage AI-powered solutions to accelerate the discovery and repurposing 
of drugs, providing affordable, timely, and accessible treatments for under-served 
populations.
Approach
 Data Collection & Preparation:
 ● Utilize bioactivity data from ChEMBL..
 ● Extract molecular structures, bioactivity metrics (e.g., IC50, EC50), and targets.
 ● Clean and preprocess data using RDKit to generate molecular descriptors
 ● Prepare molecular representations suitable for AI and GAN-based modeling (e.g., SMILES 
strings, molecular graphs).
 AI Model Development:
 ● Train regression models (XGBoost, Random Forest) for bioactivity prediction.
 ● Develop classification models to estimate drug success probability.
 ● Apply optimization techniques like GridSearchCV for model tuning.
● GAN Integration: 
○  Generator: Create novel molecular structures (SMILES strings, graphs) as new drug 
candidates.
 ○  Discriminator: Evaluate the generated molecules based on predicted bioactivity, 
drug-likeness, and similarity to known drugs.
 Molecular Visualization :
 ● SMILES Strings to Structures: Generate and display molecular structures from SMILES 
strings using  RDKit.
 ● New Drug Candidates : Visualize GAN-generated molecules alongside known compounds, 
showing bioactivity predictions and drug success estimates
Drug Repurposing:
 ● Perform *Tanimoto similarity searches* to identify potential repurposed drugs.
 ● Leverage GAN-generated molecules as additional candidates for repurposing, based on 
their similarity to existing drugs and predicted bioactivity.
 Web Interface:
 User-Friendly Interface: 
●  Input disease data and display predictions, including drug formulas and success rates.
 ●   Display molecular structures with visualizations (RDKit).
 ●   Show suggested repurposed drugs based on molecular similarity and GAN-generated 
candidates.
Benefits of GAN Integration:
 ● Novel Drug Discovery: Generate new drug-like molecules with the potential to treat 
rare diseases.
 ● Accelerated Drug Repurposing: Enhance the identification of repurposed drugs 
through molecular similarity searches, including GAN-generated molecules.
 ● Affordable Treatments: Speed up the process of drug discovery and repurposing to 
make treatments more affordable and accessible to underserved populations.
THANK YOU
