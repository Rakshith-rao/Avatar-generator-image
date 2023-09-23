# Avatar-generator-image
This repository contains code for training a Generative Adversarial Network (GAN) to generate images. The GAN consists of a generator and a discriminator, which are trained simultaneously to generate realistic images.
Prerequisites
Before running the code, make sure you have the following dependencies installed:
Python 3.x
PyTorch
NumPy
Matplotlib
You can install the required packages by running the following command:
pip install torch numpy matplotlib
Getting Started
Clone the repository:

git clone https://github.com/your-username/your-repository.git
Run the code:

python main.py
Code Structure
main.py: This is the main script that trains the GAN and saves the trained generator.
generator.py: This file contains the definition of the generator model.
discriminator.py: This file contains the definition of the discriminator model.
dataset.py: This file defines a custom dataset class for generating random noise.
utils.py: This file contains utility functions for saving generated images and visualizing the results.
Training
The GAN is trained for a specified number of epochs. During each epoch, the generator and discriminator are updated alternately to improve the quality of the generated images. The progress of the training is printed to the console, and generated images are saved at specified intervals.
Results
After training, the generator is saved as generator.pth. You can use this trained generator to generate new images by running the following code:
python

from generator import Generator
import torch

# Load the trained generator
generator = Generator(latent_dim)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# Generate a sample image
z_sample = torch.randn(1, latent_dim)
sample_image = generator(z_sample)

# Display and save the sample image
Generated Images
The generated images are saved in the generated_images directory. You can visualize the generated images by running the following code:
python

import matplotlib.pyplot as plt
import os

# Visualize some generated images
sample_images = os.listdir('generated_images')
sample_images.sort()
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
cnt = 0
for i in range(5):
    for j in range(5):
        img = plt.imread(os.path.join('generated_images', sample_images[cnt]))
        axs[i, j].imshow(img)
        axs[i, j].axis('off')
        cnt += 1
plt.show()
