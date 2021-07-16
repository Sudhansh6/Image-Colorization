import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.cuda.empty_cache()
transforms = Compose([Resize((224,224)),ToTensor()])
training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transforms
)

test_data = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transforms
)

rnd = 42
learning_rate = 2e-2
batch_size = 128
epochs = 10

img_size = 224
classes = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
      # o = (i - )
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
   )
            # 6*6*256 = 9216
            # Fully connected and Dropout layers
            
        self.denselayer = nn.Sequential(
            nn.Linear(in_features=6400, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        x = self.linear_relu_stack(x)
        # print(x.shape)
        x = self.flatten(x)
        
        x = self.denselayer(x)
        return x


train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True)

model = NeuralNetwork().to(device)
# print(model)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 5e-3, momentum = 0.9)

train_acc = []
test_acc = []

def train(dataloader, model, loss_fn, optimizer):
    correct = 0
    batch = 0
    tsize = len(dataloader.dataset)
    for X, y in tqdm(dataloader, position = 0, leave = True):
        batch += 1
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        size = batch_size
        accuracy = 100 * correct / size
        
        if batch == 1:
            train_acc.append(accuracy)

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            tqdm.write(f"loss: {loss:>3f}  accuracy: {accuracy} [{current:>5d}/{tsize:>5d}]")
            # break

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    accuracy = (100*correct)
    test_acc.append(accuracy)
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    train_acc = []
    test_acc = []

    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

plt.plot(train_acc)
plt.plot(test_acc)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")