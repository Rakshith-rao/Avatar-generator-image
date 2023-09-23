import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image

# Define the Generator class
class AvatarGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(AvatarGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, latent_codes):
        images = self.generator(latent_codes)
        images = images.view(-1, 3, 64, 64)
        return images

# Define the Discriminator class
class AvatarDiscriminator(nn.Module):
    def __init__(self):
        super(AvatarDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, images):
        probabilities = self.discriminator(images)
        return probabilities

# Function to train the GAN
def train(generator, discriminator, dataloader, epochs):
    # Define the loss function and optimizer for the generator and discriminator
    generator_loss_function = nn.BCELoss()
    discriminator_loss_function = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # Train the generator and discriminator for the specified number of epochs
    for epoch in range(epochs):
        for real_images, _ in dataloader:
            real_images = real_images.to(device)

            # Generate fake images
            latent_codes = torch.randn(real_images.size(0), generator.latent_dim).to(device)
            fake_images = generator(latent_codes)

            # Train the discriminator
            real_labels = torch.ones(real_images.size(0)).to(device)
            fake_labels = torch.zeros(fake_images.size(0)).to(device)

            real_probabilities = discriminator(real_images)
            fake_probabilities = discriminator(fake_images)

            real_loss = discriminator_loss_function(real_probabilities, real_labels)
            fake_loss = discriminator_loss_function(fake_probabilities, fake_labels)

            discriminator_loss = real_loss + fake_loss

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train the generator
            generator_labels = torch.ones(fake_images.size(0)).to(device)

            fake_probabilities = discriminator(fake_images)

            generator_loss = generator_loss_function(fake_probabilities, generator_labels)

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        # Print the generator and discriminator losses
        print('Epoch {}: Generator loss: {:.4f}, Discriminator loss: {:.4f}'.format(epoch, generator_loss.item(), discriminator_loss.item()))

# Function to generate an avatar image
def generate_avatar(generator, latent_code):
    generator.eval()  # Set the generator to evaluation mode
    fake_image = generator(latent_code.unsqueeze(0))
    generator.train()  # Set the generator back to training mode
    fake_image = fake_image.view(3, 64, 64)
    fake_image = fake_image.permute(1, 2, 0)
    fake_image = (fake_image + 1) / 2
    return fake_image

if __name__ == '__main__':
    # Define the latent dimension of the generator
    latent_dim = 128

    # Initialize the generator and discriminator
    generator = AvatarGenerator(latent_dim)
    discriminator = AvatarDiscriminator()

    # Specify the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Define the dataset path (replace with your dataset path)
    dataset_path = '/path/to/your/dataset'

    # Load the dataset and calculate mean and standard deviation values
    dataset = datasets.ImageFolder(dataset_path, transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]))

    mean = torch.stack([sample.mean(1).mean(1) for sample, _ in dataset]).mean(0)
    std = torch.stack([sample.std(1).std(1) for sample, _ in dataset]).std(0)

    # Normalize the dataset
    dataset.transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create a dataloader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the number of training epochs
    epochs = 100

    # Train the generator and discriminator
    train(generator, discriminator, dataloader, epochs)

    # Save the trained generator
    torch.save(generator.state_dict(), 'generator.pt')

    # Generate an avatar image from a latent code
    latent_code = torch.randn(1, latent_dim).to(device)
    avatar_image = generate_avatar(generator, latent_code)

    # Convert the PyTorch tensor to a PIL Image and save
    avatar_image = (avatar_image * 255).byte()  # Scale to 0-255 range
    avatar_image = transforms.ToPILImage()(avatar_image)
    avatar_image.save('avatar.png')
