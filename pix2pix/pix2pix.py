"""
    Imports
"""
import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
from torch import nn
from torchvision import datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
from dataset import ImageDataset
from models import Generator, Discriminator, init_weights

torch.cuda.empty_cache()
import warnings
warnings.filterwarnings("ignore")
"""
    Define the training loop
"""
def train(generator, discriminator, batchSize, dataloader, lambdaL1): 
    BCELoss = nn.BCEWithLogitsLoss()
    L1Loss = nn.L1Loss()
    i = 0
    fake = torch.tensor(0.1)
    real = torch.tensor(0.9)
    
    def train_discriminator(img):
        fake_color = generator(img['L'])

        discriminator.train()
        for p in discriminator.parameters():
            p.requires_grad = True
        discriminatorOptimizer.zero_grad()
        # Generate Fake images
        fake_img = torch.cat([img['L'], fake_color], dim = 1)
        fake_pred = discriminator(fake_img.detach())
        fake = torch.zeros_like(fake_pred)
        fake = torch.add(fake, 0.1)
        loss_fake = BCELoss(fake_pred, fake)

        # Generate Real images
        real_img = torch.cat([img['L'], img['ab']], dim = 1)
        real_pred = discriminator(real_img)
        real = torch.add(fake, 0.8)
        loss_real = BCELoss(real_pred, real)

        # Train the Discriminator
        loss_D = (loss_fake + loss_real)/2
        loss_D.backward()
        discriminatorOptimizer.step()
        generator.train()
        for p in discriminator.parameters():
            p.requires_grad = False
        return float(loss_fake), float(loss_real), float(loss_D), fake_color

    def train_generator(fake_color):
        # Train the Generator
        generatorOptimizer.zero_grad() 
        # fake_color = generator(img['L'])
        fake_img = torch.cat([img['L'], fake_color], dim = 1)
        fake_pred = discriminator(fake_img)
        real = torch.add(torch.zeros_like(fake_pred), 0.9)
        GAN_loss = BCELoss(fake_pred, real)
        L1_loss = L1Loss(fake_color, img['ab'])*lambdaL1
        loss_G = GAN_loss + L1_loss
        loss_G.backward()
        generatorOptimizer.step()
        return float(loss_G)

    def lab_to_rgb(L, ab):
       # x_ref=95.047, y_ref=100., z_ref=108.883
        L = (L + 1.) * 50.
        ab = ab * 110.
        Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        rgb_imgs = []
        for img in Lab:
            img_rgb = lab2rgb(img)
            rgb_imgs.append(img_rgb)
        return np.stack(rgb_imgs, axis=0)

    def plot(img):
        generator.eval()
        with torch.no_grad():
                fake_color = generator(img['L']).detach()
        fake_imgs = lab_to_rgb(img['L'], fake_color)
        real_imgs = lab_to_rgb(img['L'], img['ab'])

        plt.figure()
        for i in range(0, 5):
            ax = plt.subplot(3, 5, i + 5 + 1)
            ax.axis("off")
            plt.title('Fake')
            plt.imshow(fake_imgs[i - 1])
            ax = plt.subplot(3, 5, i + 1)
            ax.axis("off")
            plt.title('Real')
            plt.imshow(real_imgs[i - 1])
            ax = plt.subplot(3, 5, i + 10 + 1)
            ax.axis("off")
            plt.title('Grayscale')
            plt.imshow(img['L'][i - 1][0].cpu(), cmap = 'gray')
        plt.show()

    for img in tqdm(dataloader, position=0, leave=True):
        i += 1
        img['L'] = img['L'].to(device)
        img['ab'] = img['ab'].to(device)
        floss, rloss, loss_D, fake_color = train_discriminator(img)
        loss_G = train_generator(fake_color)
        if i % 250 == 1:
            print(f"\nIteration {i}/{len(train_dataloader)}", end = "     ")
            print(f"DLoss = {loss_D}, GLoss = {loss_G}")
            print(f"Real Loss: {rloss}, Fake Loss: {floss}")
        
        if i % 250 == 1:
            plot(img)

"""
    Defining the models and the network parameters
"""
# torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
 
# Models
n_down = 6
generator = Generator(input_c=1, output_c=2, n_down=n_down, num_filters=64).to(device)
discriminator = Discriminator(3).to(device)
generator = init_weights(generator)
discriminator = init_weights(discriminator)
 
# Parameters
size = 2**n_down
rnd = 4
lrD = 2e-4
lrG = 2e-4
batchSize = 16
nEpochs = 100
lambdaL1 = 100.

# Optimizers
generatorOptimizer = torch.optim.Adam(generator.parameters(), lr=lrG, betas=(0.5, 0.999))
discriminatorOptimizer = torch.optim.Adam(discriminator.parameters(), lr=lrD, betas=(0.5, 0.999))

"""
    Loading the dataset 
"""
trainDataset = ImageDataset('train', size)
train_dataloader = DataLoader(trainDataset, batch_size=batchSize, num_workers=4, pin_memory=True)

"""
Calling the training loop
"""
for e in range(nEpochs):
    train(generator, discriminator, batchSize, train_dataloader, lambdaL1)
    torch.save(generator.state_dict(), "generator.pth")
    print("Saved Generator Model State to generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Saved Discriminator Model State to discriminator.pth")