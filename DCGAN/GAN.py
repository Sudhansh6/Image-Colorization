"""
  Imports
"""
import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
import matplotlib.pyplot as plt
import numpy as np
LatentDim = 128
"""
  Neural Network for the Discriminator
"""
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.classifier = nn.Sequential(
         # o = (i + 2p - k)/s + 1
            nn.Conv2d(1, 64, kernel_size = 3, stride = 2, bias = False), # 13
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2, bias = False), # 6
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
    self.dense = nn.Sequential(
            nn.Linear(36*64, 1),
            nn.Sigmoid()
        )
  def forward(self, x):
    x = self.classifier(x)
    x = torch.flatten(x, 1)
    x = self.dense(x)
    return x
 
"""
  Neural Network for the Generator
"""
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.dense = nn.Sequential(
        nn.Linear(LatentDim, 6*6*128),
        nn.ReLU()
      )
    # o = (i + 2p - k)/s + 1
    # 7 = (16 - 4)/2 + 1 -> 6 = 2p?
    self.convolve = nn.Sequential(
        nn.ConvTranspose2d(128, 128, kernel_size = 4, stride = 2, bias = False),
        nn.BatchNorm2d(128),
        nn.ReLU(), # 16
        nn.ConvTranspose2d(128, 128, kernel_size = 4, stride = 2, bias = False),
        nn.BatchNorm2d(128),
        nn.ReLU(), # 34
        nn.Conv2d(128, 1, kernel_size = 3) # 28
      )
  def forward(self, x):
    x = self.dense(x)
    x = x.reshape(x.shape[0], 128, 6, 6)
    x = self.convolve(x)
    return x
 
"""
  Function to generate random inputs for the Generator
"""
def GeneratorInput(dim, nSamples):
  z = torch.randn(dim*nSamples).to(device)
  z = z.reshape((nSamples, dim))
  return z 
 
"""
  Function to generate output from the generator
  This acts as an input to the discriminator
"""
def FakeDiscriminatorInput(GeneratorModel, dim, nSamples, imgSize = 28, random = False):
  if random:
    x = torch.rand(imgSize*imgSize*nSamples).to(device)
    x = x.reshape((nSamples, 1, imgSize, imgSize))
    x = x.type(torch.float32)
  else:
    GeneratorModel.eval()
    z = GeneratorInput(dim, nSamples)
    x = GeneratorModel(z).to(device)
  y = torch.zeros((nSamples, 1)).to(device)
  return x.to(device), y
 
"""
  Function to sample real data for the discriminator
"""
def RealDiscriminatorInput(dataset, nSamples, s, e):
  ix = np.random.randint(0, dataset.shape[0], nSamples)
  x = dataset[s:e]
  x = x.reshape((nSamples, 1, imgSize, imgSize))
  x = x.type(torch.float32)
  y = torch.ones((nSamples, 1))
  return x.to(device), y.to(device)
 
"""
  Plot the results
"""
def plot(generator, dim, size, s):
  nSamples = size**2
  x, _ = FakeDiscriminatorInput(generator, dim, nSamples)
  x = x.cpu().data.numpy()
  x = x.reshape((nSamples, 28, 28))
  plt.figure(figsize= (s,s))
  for i in range(size):
    for j in range(size):
      ix = i*size + j
      plt.subplot(size, size, ix + 1)
      plt.imshow(x[ix], cmap="gray")
      # plt.title(y[ix])
  plt.show()
  x = generator(z_const).cpu().data.numpy()
  x = x.reshape((28,28))
  plt.figure(figsize = (s,s))
  plt.title("z const")
  plt.imshow(x, cmap="gray")
  plt.show()

"""
  Define the training loop
"""
def train(generator, discriminator, nEpochs, batchSize, LatentDim, dataset): 
    HalfBatch = batchSize//2
    nBatches = 60000//HalfBatch
    for i in tqdm(range(nEpochs), position = 0, leave = True):
      realScore, fakeScore, count = float(0), 0, 0
      for j in tqdm(range(nBatches), position = 0, leave = True):
        # zero the gradients on each iteration
        
        # Discriminator Training
        for k in range(1):
          discriminatorOptimizer.zero_grad()
          xFake, yFake = FakeDiscriminatorInput(generator, LatentDim, HalfBatch)

          # Generate examples of real data
          xReal, yReal = RealDiscriminatorInput(dataset, HalfBatch, j*HalfBatch, (j+1)*HalfBatch)

          # Train the discriminator on the true/generated data
          predReal = discriminator(xReal)
          discRealLoss = loss(predReal, yReal)

          # add .detach() here think about this
          predFake = discriminator(xFake.detach())
          discFakeLoss = loss(predFake, yFake)
          
          DLoss = (discRealLoss + discFakeLoss)/2
          DLoss.backward()
          discriminatorOptimizer.step()

        # Generator Training
        for k in range(1):
          generatorOptimizer.zero_grad()
          # Predict using Generator
          xFake, yFake = FakeDiscriminatorInput(generator, LatentDim, HalfBatch)  
          yReal = torch.ones(HalfBatch, 1).to(device)
    
          # Train the generator, We invert the labels here 
          predFake = discriminator(xFake)
          GLoss = loss(predFake, yReal)
          GLoss.backward()
          generatorOptimizer.step()

      if i % 10 == 9:
        plot(generator, LatentDim, 2, 10)
      
      print(f"\nEpoch: {i} Loss D.: {DLoss}", end = " ")
      print(f"Loss G.: {GLoss}")
      print(f"Real Score: {discRealLoss}", end = " ")
      print(f"Fake Score: {discFakeLoss}")

"""
  Pre-train Discriminator for headstart
"""
def pretrain(generator, discriminator, nEpochs, batchSize, dataset):
  HalfBatch = batchSize//2
  nBatches = 60000//HalfBatch
  for i in tqdm(range(nEpochs)):
    for j in tqdm(range(nBatches)):
      discriminatorOptimizer.zero_grad()

      xFake, yFake = FakeDiscriminatorInput(generator, LatentDim, HalfBatch, random = True)
      xReal, yReal = RealDiscriminatorInput(dataset, HalfBatch, j*HalfBatch, (j+1)*HalfBatch)

      predReal = discriminator(xReal)
      discriminatorLoss = loss(predReal, yReal)

      predFake = discriminator(xFake.detach())
      DLoss = loss(predFake, yFake)
      
      DLoss = (discriminatorLoss + DLoss)/2
      DLoss.backward()
      discriminatorOptimizer.step()

    print(f"Pretraining Epoch: {i} Loss D.: {DLoss}")
        
"""
  Defining the models and the network parameters
"""
# torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
 
# Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
 
# Parameters
imgSize = 28
rnd = 42
learningRate = 1e-4
batchSize = 64
nEpochs = 20
z_const =  torch.randn(LatentDim).to(device)
z_const = z_const.reshape((1, LatentDim))

# Optimizers
generatorOptimizer = torch.optim.Adam(generator.parameters(), lr=learningRate, betas=(0.5, 0.999))
discriminatorOptimizer = torch.optim.Adam(discriminator.parameters(), lr=learningRate, betas=(0.5, 0.999))

# loss
loss = nn.BCELoss()

"""
  Loading the MNIST dataset 
"""
transforms = Compose([Resize((imgSize, imgSize)),ToTensor()])
trainingData = datasets.MNIST(
  root="../data",
  train=True,
  download=True,
  transform=transforms
)
dataset = trainingData.data
 
"""
Calling the training loop
"""
import os
if os.path.isfile('discriminator.pth'):
  generator.load_state_dict(torch.load('generator.pth'))
  discriminator.load_state_dict(torch.load('discriminator.pth'))
train(generator, discriminator, nEpochs, batchSize, LatentDim, dataset)

"""
Saving the models
"""
torch.save(generator.state_dict(), "generator.pth")
print("Saved Generator Model State to generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
print("Saved Discriminator Model State to discriminator.pth")
