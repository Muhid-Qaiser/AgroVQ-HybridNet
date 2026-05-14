# AgroVQ-HybridNet Implementation

This repository contains the PyTorch implementation of **AgroVQ-HybridNet**, an architecture leveraging a Vector Quantized-Variational Autoencoder (VQ-VAE) combined with a latent classifier, purposed for agricultural image processing and classification tasks.

## 🚀 Pre-trained Weights

Pre-trained checkpoint weights of the pipeline can be downloaded from: 
[Google Drive Checkpoints Link](https://drive.google.com/drive/folders/1SFvYb3vsKKxPbLSIVG-sR-3Xp0lK19Fr?usp=sharing)

Extract the downloaded weights into the `checkpoints/` directory to use them for inference or fine-tuning.

## 📁 Repository Structure

```text
.
├── base.py                   # Base classes for VAE models
├── train_classifier.ipynb    # Notebook for training the latent classifier
├── train_vqvae.ipynb         # Notebook for training the core VQ-VAE model
├── types_.py                 # Type definitions and hints
├── vq_vae.py                 # Core VQ-VAE architecture (Encoder, Quantizer, Decoder)
├── vq_vae.yaml               # Model configuration and hyperparameter settings
├── checkpoints/              # Directory to store model weights (.pth files)
├── enhanced_images/          # Output directory for reconstructed/enhanced images
└── images/                   # Directory for dataset/input images

## 🧠 Model Architecture
The VQVAE implemented in vq_vae.py features:

- Encoder: Convolutional layers with LeakyReLU activations followed by residual blocks to encode images into a latent space.
- Vector Quantizer: Maps the continuous encoded outputs to the nearest discrete embeddings using an embedding dictionary.
- Decoder: Symmetrical to the encoder, with transposed convolutions to reconstruct the image from the quantized latent space.

## ⚙️ Usage
1. Training the VQ-VAE
To train the base Vector Quantized-Variational Autoencoder, open and run the cells in train_vqvae.ipynb. The training configuration (like embedding dimensions, batch size, and learning rate) is managed via the vq_vae.yaml file.

2. Training the Classifier
Once the VQ-VAE is trained (or using the provided pre-trained weights), you can train the downstream classifier on the quantized latent representations by running train_classifier.ipynb.

## 📦 Requirements
Python 3.8+
PyTorch
Torchvision
PyYAML
To install the standard deep learning environment, you can run:
pip install torch torchvision pyyaml



