import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a directory to save generated images
os.makedirs("generated_images", exist_ok=True)

# Hyperparameters
latent_dim = 100
lr = 0.0002
batch_size = 64
epochs = 100

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 64, 64)
        return img

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        validity = self.model(img)
        return validity

# Initialize generator and discriminator
generator = Generator(latent_dim)
discriminator = Discriminator()

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Transform to normalize images between -1 and 1
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class RandomNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(3, 64, 64)

# Create dataloader
dataloader = DataLoader(RandomNoiseDataset(), batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        valid = torch.ones((imgs.size(0), 1))
        fake = torch.zeros((imgs.size(0), 1))

        # Configure input
        real_imgs = imgs
        valid = valid
        fake = fake

        optimizer_D.zero_grad()

        # Loss on real images
        real_loss = adversarial_loss(discriminator(real_imgs.view(real_imgs.size(0), -1)), valid)
        # Loss on fake images
        z = torch.randn(imgs.size(0), latent_dim)
        gen_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(gen_imgs.view(gen_imgs.size(0), -1).detach()), fake)
        # Total discriminator loss
        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_D.step()


        optimizer_G.zero_grad()

        # Loss for generator (maximizing the discriminator's loss)
        g_loss = adversarial_loss(discriminator(gen_imgs.view(gen_imgs.size(0), -1)), valid)

        g_loss.backward()
        optimizer_G.step()

        # Print progress
        print(
            f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save generated images at specified intervals
        if epoch % 10 == 0 and i == 0:
            save_image(gen_imgs.data[:25], f"generated_images/{epoch}_{i}.png", nrow=5, normalize=True)

# Save the trained generator
torch.save(generator.state_dict(), 'generator.pth')

# Generate a sample image from the trained generator
z_sample = torch.randn(1, latent_dim)
sample_image = generator(z_sample)
sample_image = (sample_image + 1) / 2.0  # Denormalize to [0, 1] range

# Display and save the sample image
plt.imshow(sample_image.squeeze().permute(1, 2, 0).detach().numpy())
plt.axis('off')
plt.show()
save_image(sample_image, 'sample_generated_image.png', normalize=False)
