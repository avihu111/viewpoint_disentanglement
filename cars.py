import os
import numpy as np
from torchvision.datasets import VisionDataset
import torch
from PIL import Image


class Cars3D(VisionDataset):
	def __init__(self, root, transform=None, target_transform=None, return_indices=True):
		super().__init__(root, transform=transform, target_transform=target_transform)
		self.__data_path = os.path.join(root, 'cars3d.npz')
		self.imgs = np.load(self.__data_path)['imgs']
		self.classes = np.zeros(shape=(self.imgs.shape[0], ), dtype=np.long)

		self.contents = np.zeros(shape=(self.imgs.shape[0], ), dtype=np.long)

		for elevation in range(4):
			for azimuth in range(24):
				for object_id in range(183):
					img_idx = elevation * 24 * 183 + azimuth * 183 + object_id
					self.classes[img_idx] = object_id
					self.contents[img_idx] = elevation * 24 + azimuth
		self.return_indices = return_indices
		self.num_classes = len(np.unique(self.classes))
		self.targets = self.classes

	def __getitem__(self, item):
		img = self.imgs[item]
		# new_img = torch.Tensor(img).permute(2,1,0) / 255.
		new_img = Image.fromarray(img)
		new_img = self.transform(new_img)
		cls = self.classes[item]
		if self.return_indices:
			return new_img, cls, item
		return new_img

	def __len__(self):
		return len(self.imgs)
