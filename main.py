from notebooks.data_preparation import fetch_data
from notebooks.model_training import train_models
from notebooks.gan_training import train_gan

# Fetch and preprocess data
fetch_data('CHEMBL203')

# Train models
train_models()

# Train GAN
train_gan()