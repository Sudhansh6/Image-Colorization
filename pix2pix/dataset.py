"""
	Import Files 
"""
import pickle
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
import numpy as np

"""
	Class to generate datasets
"""
class ImageDataset(Dataset):
	def __init__(self, dtype, SIZE):
		if dtype == 'train':
			self.transforms = transforms.Compose([
				transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
				transforms.RandomHorizontalFlip(), # A little data augmentation!
			])
			file = open("mini-imagenet-cache-train.pkl", "rb")
			arr = pickle.load(file)
			self.paths = arr["image_data"].reshape([64*600, 84, 84, 3])

		elif dtype == 'val':
			self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
			file = open("mini-imagenet-cache-test.pkl", "rb")
			arr = pickle.load(file)
			self.paths = arr["image_data"].reshape([20*600, 84, 84, 3])

		self.dtype = dtype
		self.size = SIZE
		
	def __getitem__(self, idx):
		img = Image.fromarray(np.uint8(self.paths[idx])).convert("RGB")
		img = self.transforms(img)
		img = np.array(img)
		imgLab = rgb2lab(img).astype('float32')
		imgLab = transforms.ToTensor()(imgLab)
		L = imgLab[[0], ...]/50. - 1.
		ab = imgLab[[1, 2], ...]/110. 
		return {'L': L, 'ab': ab}

	def __len__(self):
		return self.paths.shape[0]
