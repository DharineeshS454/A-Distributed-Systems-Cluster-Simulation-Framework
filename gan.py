import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 100
IMAGE_SIZE = 64
LEARNING_RATE = 0.0002
BETA1 = 0.5
NUM_EPOCHS = 100

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size: 512 x 4 x 4
            
            # Using Upsample + Conv2d instead of ConvTranspose2d to reduce checkerboard artifacts
            nn.Upsample(scale_factor=2, mode='nearest'), # Upsample to 8x8
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),    # Conv to keep 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size: 256 x 8 x 8
            
            nn.Upsample(scale_factor=2, mode='nearest'), # Upsample to 16x16
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),    # Conv to keep 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: 128 x 16 x 16
            
            nn.Upsample(scale_factor=2, mode='nearest'), # Upsample to 32x32
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),     # Conv to keep 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size: 64 x 32 x 32
            
            nn.Upsample(scale_factor=2, mode='nearest'), # Upsample to 64x64
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),      # Conv to keep 64x64
            nn.Tanh()
            # Final size: 3 x 64 x 64
        )

    def forward(self, x):
        return self.main(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 64 x 32 x 32
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 128 x 16 x 16
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 256 x 8 x 8
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 512 x 4 x 4
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Initialize networks
generator = Generator(LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.02.
generator.apply(weights_init)
discriminator.apply(weights_init)

# Loss function and optimizers
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# Training function
def train_gan(dataloader):
    for epoch in range(NUM_EPOCHS):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
        
        # Save generated images
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_images = generator(torch.randn(16, LATENT_DIM, 1, 1).to(device))
                fake_images = fake_images.cpu().detach()
                plt.figure(figsize=(10, 10))
                for j in range(16):
                    plt.subplot(4, 4, j+1)
                    plt.imshow(np.transpose(fake_images[j], (1, 2, 0)))
                    plt.axis('off')
                plt.savefig(f'generated_images_epoch_{epoch+1}.png')
                plt.close()

# Example usage
if __name__ == "__main__":
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load your dataset here
    # dataset = torchvision.datasets.ImageFolder(root='path_to_your_dataset', transform=transform)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Uncomment the following line to start training
    # train_gan(dataloader) 