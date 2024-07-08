from PIL import Image
import torch
from torch import Tensor
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import sys
from training import *
from training_2 import * 

sys.path.append('C:/Users/zacca/Python/Deep_Learning_modules/Torch_Models')
from dcgan import *

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB') 

        if self.transform:
            img = self.transform(img)

        return img

# Create a transform to resize the data
image_size = 128
transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_dataset(data_path, batch_size=64, workers = 2):
    images = torch.load(data_path)
    train_dataset = CustomDataset(images, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=workers) 

    return train_dataloader

if __name__ == '__main__':
    # Initialize generator and discriminator
    latent_dim = 100
    generator = Generator(latent_dim, image_size)
    discriminator = Discriminator(image_size)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


    data_path = 'D:/Datasets/Images/CelebA/face_dataset.pth'
    #Dataset =  get_dataset(data_path)
    Dataset =  get_dataset(data_path, batch_size = 256)

    train_model(generator, discriminator, Dataset)


'''
Ways to Address Dominate Discriminator:

    - Reduce the learning rate of the discriminator .0002 -> .0001
    - Increase the number of generator updates per discriminator update.
    - Apply label smoothing to the real labels in the discriminator training 
    - Add random noise to the labels used for training the discriminator
    - Add a gradient penalty to the discriminator loss
    - Using a larger batch size
    - increase complexity of generator


Other improvments:
    - feature matching
    - Mini-Batch Discrimination
    - Historical Averaging
    - Virtual Batch Normalization
    - improve dis to match gen

'''