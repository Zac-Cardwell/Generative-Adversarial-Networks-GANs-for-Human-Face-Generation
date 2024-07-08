import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from torchvision.utils import save_image


'''Edit the name of the save file for all the information'''
save_name = "comp_lr_noise"



def combined_loss(discriminator, real_images, fake_images, real_labels, fake_labels, adv_weight=1.0, fm_weight=10.0):
    # Get validity and features from discriminator
    real_validity, real_features = discriminator(real_images, feature=True)
    fake_validity, fake_features = discriminator(fake_images, feature=True)

    # Standard adversarial loss
    adversarial_loss = F.binary_cross_entropy(real_validity, real_labels) + \
                       F.binary_cross_entropy(fake_validity, fake_labels)

    # Feature matching loss
    feature_matching_loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        feature_matching_loss += torch.mean((real_feat - fake_feat) ** 2)

    # Normalize feature matching loss
    feature_matching_loss /= len(real_features)

    # Combine losses with weights
    total_loss = adv_weight * adversarial_loss + fm_weight * feature_matching_loss
    return total_loss


def make_plot(g_loss, d_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(g_loss, label='Generator')
    plt.plot(d_loss, label='Discriminator')

    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over 100 Epochs')

    plt.legend(loc='upper right', fontsize=10)

    # Adjust subplot parameters to make space for the text
    plt.subplots_adjust(bottom=0.2)

    # Add description below the plot
    description = "Losses are taken every 100 batches, with a batch size of 64 and a total of 3166 batches per epoch"\
    "G_lr: .0001, D_lr: .0003, loss_func: combined loss, Noise: True, Label_smoothing: True, Models: modified"
    plt.figtext(0.5, 0.02, description, ha='center', fontsize=10, wrap=True)

    # Save the plot to a file
    plt.savefig(f'Faces_1.0/losses/{save_name}_training_loss.png')


# Define the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model(generator, discriminator, data_loader):

    G_losses = []
    D_losses = []

    lr = 0.0001
    num_epochs  = 100
    latent_dim = 100

    fixed_noise = torch.randn(64, 100).to(device)

    # Initialize the generator and discriminator
    generator.to(device)
    discriminator.to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    '''Learning rate for the discriminator'''
    #optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))  # Reduced learning rate

    #Loss Function
    adversarial_loss = nn.BCELoss().to(device)



    for epoch in range(num_epochs):
        i=0
        for real_images in tqdm(data_loader):

            real_images = real_images.to(device)
            z = torch.randn(real_images.size(0), latent_dim).to(device)

            # Generate a batch of images
            fake_images = generator(z)


            ############################
            # (1) Update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            discriminator.train()
            optimizer_D.zero_grad()

            # Create labels for real and fake image batches of size (batch_size, 1)
            '''Add noise to the labels'''
            real_labels = torch.ones(real_images.size(0), 1).to(device)  * (0.9 + 0.1 * torch.rand(real_images.size(0), 1).to(device)) # Random noise
            fake_labels = torch.zeros(real_images.size(0), 1).to(device) * (0.1 * torch.rand(real_images.size(0), 1).to(device))

            # get the discriminator outputs for real and fake images
            real_outputs = discriminator(real_images)
            d_loss_real = adversarial_loss(real_outputs, real_labels)
            d_loss_real.backward()

            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = adversarial_loss(fake_outputs, fake_labels)
            d_loss_fake.backward()

            d_loss = (d_loss_real + d_loss_fake)/2
            #d_loss.backward()
            optimizer_D.step()



            ############################
            # (2) Update Generator: maximize log(D(G(z)))
            ###########################
            generator.train()

            '''Increase the number of generator updates per discriminator update.'''
            #for _ in range(2):
            optimizer_G.zero_grad()

            # Generate a batch of images again
            #z = torch.randn(real_images.size(0), 100).to(device)
            #fake_images = generator(z)

            # Since we just updated D, perform another forward pass of all-fake batch through D
            fake_outputs = discriminator(fake_images)

            '''Feature Matching'''
            g_loss = adversarial_loss(fake_outputs, real_labels)  # Use real labels to fool the discriminator
            #g_loss = combined_loss(discriminator, real_images, fake_images, real_labels, fake_labels)

            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                # Save Losses for plotting later
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())

            i += 1



        print(f"Epoch [{epoch}/{num_epochs}]"
            f"Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}\n"
            f"Discriminator Real Loss: {d_loss_real.item():.4f}, Discriminator Fake Loss: {d_loss_fake.item():.4f}")

        # Save generated images on fixed noise 
        if (epoch + 1) % 5 == 0 or epoch == 0:
            generator.eval()
            with torch.no_grad():
                Val_img = generator(fixed_noise)
                save_image(Val_img, f"Faces_1.0/gen_images/{save_name}_{epoch+1}.png", nrow=8, normalize=True)

        # Save model checkpoints
        if (epoch + 1) % 25 == 0 or epoch == 0:
            torch.save(generator.state_dict(), f"Faces_1.0/models/generator_{save_name}_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"Faces_1.0/models/discriminator_{save_name}_{epoch + 1}.pth")

        torch.save(G_losses, f'Faces_1.0/losses/{save_name}_generator_losses.pth')
        torch.save(D_losses, f'Faces_1.0/losses/{save_name}_discriminator_losses.pth')
        make_plot(G_losses, D_losses)



