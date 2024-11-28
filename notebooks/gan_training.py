import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset

# File path to the dataset
data_file_path = 'S:\\vertex\\Final\\data\\processed_data.csv'

# Check if the file exists
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"The file at '{data_file_path}' was not found. Please check the file path.")

# Load the real dataset (ensure it is numeric)
try:
    real_data = pd.read_csv(data_file_path)
except Exception as e:
    raise RuntimeError(f"An error occurred while reading the CSV file: {str(e)}")

# Check if the dataset contains the necessary columns
required_columns = ['SMILES', 'MolecularWeight', 'LogP', 'Activity']
missing_columns = [col for col in required_columns if col not in real_data.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing in the dataset: {', '.join(missing_columns)}")

# Drop the 'SMILES' column as it's non-numeric and irrelevant for training
real_data = real_data.drop(columns=['SMILES'])

# Handle missing values by removing rows with NaN values in the 'Activity' column
real_data = real_data.dropna(subset=['Activity'])

# Check the shape of the dataset
print(f"Real Data Shape: {real_data.shape}")
print(f"First few rows of data:\n{real_data.head()}")

# Ensure the dataset contains only numeric values for GAN training
real_data = real_data.select_dtypes(include=[np.number])  # Only numerical columns
print(f"Filtered Data Shape: {real_data.shape}")

# Check if the dataset is empty
if real_data.shape[0] == 0:
    raise ValueError("The real dataset is empty. Cannot proceed with training.")

# Convert the dataset into a tensor
real_data_tensor = torch.tensor(real_data.values, dtype=torch.float32)

# Check if the tensor is empty
print(f"Real Data Tensor Shape: {real_data_tensor.shape}")
if real_data_tensor.size(0) == 0:
    raise ValueError("Tensor is empty. Cannot continue training.")

# Training parameters
z_dim = 100  # Dimensionality of the latent vector (input to generator)
batch_size = 64  # Size of each batch
lr = 0.0002  # Learning rate for the optimizers
epochs = 50  # Number of training epochs

# Modify Generator and Discriminator for 3D input (adjusting for 3 features in the real dataset)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),  # Input latent vector (z_dim) with 128 features
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3),  # Output 3 features (matching the input data dimensionality)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 512),  # Input layer for 3 features from the real data (MolecularWeight, LogP, Activity)
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),  # Output a single probability (real or fake)
            nn.Sigmoid(),  # Sigmoid ensures output is between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

# Initialize models and optimizers
generator = Generator()
discriminator = Discriminator()
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
loss_function = nn.BCELoss()  # Binary Cross-Entropy loss function

# Create a dataset and dataloader
real_dataset = TensorDataset(real_data_tensor)
dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)  # Ensure shuffling each epoch

# Training loop with DataLoader
for epoch in range(epochs):
    for real_data_batch in dataloader:  # Iterate over full dataset in batches
        real_data_batch = real_data_batch[0]  # Extract the tensor from the tuple returned by DataLoader
        
        # Ensure the batch size is correct in the case of the last smaller batch
        current_batch_size = real_data_batch.size(0)

        # Generate random z values (latent space)
        z = torch.randn(current_batch_size, z_dim)  # Latent vector from a random normal distribution
        fake_data = generator(z)  # Generate fake data using the generator

        # Real and fake labels (real data = 1, fake data = 0)
        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)

        # Train Discriminator
        d_optimizer.zero_grad()  # Zero the gradients before backprop
        real_loss = loss_function(discriminator(real_data_batch), real_labels)  # Loss for real data
        fake_loss = loss_function(discriminator(fake_data.detach()), fake_labels)  # Loss for fake data
        d_loss = real_loss + fake_loss  # Total discriminator loss
        d_loss.backward()  # Backpropagate the discriminator loss
        d_optimizer.step()  # Update the discriminator's weights

        # Train Generator
        g_optimizer.zero_grad()  # Zero the gradients before backprop
        g_loss = loss_function(discriminator(fake_data), real_labels)  # Generator's loss (trying to fool discriminator)
        g_loss.backward()  # Backpropagate the generator loss
        g_optimizer.step()  # Update the generator's weights

    # Print loss statistics at the end of each epoch
    print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Save the models after training
try:
    torch.save(generator.state_dict(), 'S:\\vertex\\Final\\models\\gan_generator.pth')
    torch.save(discriminator.state_dict(), 'S:\\vertex\\Final\\models\\gan_discriminator.pth')
    print("GAN training complete. Models saved.")
except Exception as e:
    print(f"An error occurred while saving the models: {str(e)}")
