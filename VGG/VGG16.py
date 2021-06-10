import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

img_size = 64
transforms = Compose([Resize((img_size, img_size)),ToTensor()])
training_data = datasets.MNIST(
	root="data",
	train=True,
	download=True,
	transform=transforms
)

test_data = datasets.MNIST(
	root="data",
	train=False,
	download=True,
	transform=transforms
)

rnd = 420
learning_rate = 0.05
batch_size = 64
epochs = 7
classes = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.features = nn.Sequential(
          # o = (i - k + 2*p)/s + 1
          # o = (i - p + 2*p)/s + 1
					nn.Conv2d(1, 64, kernel_size=3, padding=1), # 224 + 2 - 3 + 1 = 224/64
					nn.ReLU(inplace=True),
					nn.Conv2d(64, 64, kernel_size=3, padding=1), # 224  + 2 - 3 + 1 = 224
					nn.ReLU(inplace=True),
					nn.MaxPool2d(kernel_size=2, stride=2), # (224 - 2)/2 + 1 = 112/32

					nn.Conv2d(64, 128, kernel_size=3, padding=1), # 112
					nn.ReLU(inplace=True),
					nn.Conv2d(128, 128, kernel_size=3, padding=1), # 112
					nn.ReLU(inplace=True),
					nn.MaxPool2d(kernel_size=2, stride=2), # 56/16

					# nn.Conv2d(128, 256, kernel_size=3, padding=1), # 56
					# nn.ReLU(inplace=True),
					# nn.Conv2d(256, 256, kernel_size=3, padding=1), # 56
					# nn.ReLU(inplace=True),
					# nn.Conv2d(256, 256, kernel_size=3, padding=1), # 56
					# nn.ReLU(inplace=True),
					# nn.MaxPool2d(kernel_size=2, stride=2), # 28/8

					# nn.Conv2d(256, 512, kernel_size=3, padding=1), 
					# nn.ReLU(inplace=True),
					# nn.Conv2d(512, 512, kernel_size=3, padding=1),
					# nn.ReLU(inplace=True),
					# nn.Conv2d(512, 512, kernel_size=3, padding=1),
					# nn.ReLU(inplace=True),
					# nn.MaxPool2d(kernel_size=2, stride=2), # 14/4

					# nn.Conv2d(512, 512, kernel_size=3, padding=1),
					# nn.ReLU(inplace=True),
					# nn.Conv2d(512, 512, kernel_size=3, padding=1),
					# nn.ReLU(inplace=True),
					# nn.Conv2d(512, 512, kernel_size=3, padding=1),
					# nn.ReLU(inplace=True),
					# nn.MaxPool2d(kernel_size=2, stride=2) # 7/2
    )
    self.avgpool = nn.AdaptiveAvgPool2d(output_size = (7, 7))
    self.classifier = nn.Sequential(
					nn.Linear(49*128, 4096),
					nn.ReLU(inplace=True),
					nn.Dropout(p = 0.5),
					# nn.Linear(4096, 4096),
					# nn.ReLU(inplace=True),
					# nn.Dropout(p = 0.5),
          nn.Linear(4096, 10),
          nn.Softmax()
			)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.size())
    x = self.classifier(x)
    return x

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 5e-4)

train_acc = []
test_acc = []

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  correct = 0
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

		# Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

		# Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>3f}	[{current:>5d}/{size:>5d}]")
  print(correct, size)
  accuracy = 100 * correct / size
  return accuracy

def test(dataloader, model):
	size = len(dataloader.dataset)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
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
  print(f"Epoch {t+1}\n-------------------------------")
  train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
  train_acc.append(train_accuracy)
  test_accuracy = test(test_dataloader, model)
  test_acc.append(test_accuracy)
print("Done!")

print(train_acc)
plt.plot(train_acc)
plt.plot(test_acc)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")