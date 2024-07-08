import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self,latent_dim, image_size, channels=3, kernel_size=4):
        super(Generator, self).__init__()
        '''
        args:
            latent_dim - Size of the input noise vector z
            image_size - the size of the images in the dataset
            opt_channels - number of desired output channels 
            kernel_size - size of kernel

        init_size -  inatial size of image to be gradually increased as it passes through the network.
            the goal is to start with a low-resolution representation of an image and gradually upsample it
            to generate a high-resolution image.
        linear1 - takes the input vector z and outputs a vector to be reshaped into size 
            (batch_size x 128 x init_size x init_size)

        each convtrans layer multiplies the input_size by 2 , with 4 layers = 2^5 = 32 so we divide the 
            desired output image_size by 32 in  init_size 
        '''
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.opt_channels = channels
        self.kernel_size = kernel_size


        self.init_size = self.image_size // 32
        self.linear1 = nn.Sequential(nn.Linear(in_features = self.latent_dim, out_features = 512 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=self.opt_channels, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.linear1(z)
        #print(out.size())
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        #print(out.size())
        img = self.conv_blocks(out)
        #print(img.size())
        return img



class Discriminator(nn.Module):
    def __init__(self, image_size, input_channels=3):
        super(Discriminator, self).__init__()
        '''
        args:
            image_size = size of input image
            input_channels = number of channels in input image 

        The discriminator is just a normal CNN binary classifier 
        '''
        self.image_size = image_size
        self.input_channels = input_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(256)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(512)
        )

        # The height and width of downsampled image
        ds_size = self.image_size // 16
        self.adv_layer = nn.Sequential(spectral_norm(nn.Linear(512 * ds_size ** 2, 1)), nn.Sigmoid())


    def forward(self, img, feature = False):
        out1 = self.conv1(img)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out_flat = out4.view(out4.shape[0], -1)
        validity = self.adv_layer(out_flat)

        if feature:
            return validity, [out1, out2, out3, out4]
            
        return validity



if __name__ == '__main__':

    latent_dim = 100
    image_size = 64

    generator = Generator(latent_dim, image_size)
    discriminator = Discriminator(image_size)

    z = torch.randn(64, 100)

    fake_images = generator(z)
    print(fake_images.size())
    fake_outputs = discriminator(fake_images.detach())
    print(fake_outputs.size())
