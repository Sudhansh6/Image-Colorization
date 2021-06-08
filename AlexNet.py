import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
transforms = Compose([Resize((227,227)),ToTensor()])
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

rnd = 42
learning_rate = 0.005
batch_size = 6000
epochs = 10

img_size = 227
classes = 10

subset = list(range(0, len(training_data), 10))
training_data = Subset(training_data, subset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			# 6*6*256 = 9216
			# Fully connected and Dropout layers
            nn.Flatten(),
			nn.Dropout(p=0.5),
			nn.Linear(in_features=9216, out_features=4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(in_features=4096, out_features=1024),
			nn.ReLU(),
			nn.Linear(in_features=1024, out_features=10),
			nn.Softmax()
		)

	def forward(self, x):
		# x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits


train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay = 5e-3)

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

		if batch == 0:
			accuracy = 100 * correct / size
			train_acc.append(accuracy)

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			accuracy = 100 * correct / size
			print(f"loss: {loss:>3f}  [{current:>5d}/{size:>5d}]")

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