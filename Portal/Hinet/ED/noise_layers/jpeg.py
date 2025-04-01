import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import random, string


class JpegTest(nn.Module):
	def __init__(self, Q=50, subsample=2, path="temp/"):
		super(JpegTest, self).__init__()
		self.Q = Q
		self.subsample = subsample
		self.path = path
		if not os.path.exists(path): os.mkdir(path)
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

	def get_path(self):
		return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

	def forward(self, image_cover_mask):
		image, cover_image = image_cover_mask[0], image_cover_mask[1]

		noised_image = torch.zeros_like(image)

		for i in range(image.shape[0]):
			single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
			im = Image.fromarray(single_image)

			file = self.get_path()
			while os.path.exists(file):
				file = self.get_path()

			im.save(file, format="JPEG", quality=self.Q, subsampling=self.subsample)
			jpeg = np.array(Image.open(file), dtype=np.uint8)
			os.remove(file)

			noised_image[i] = self.transform(jpeg).unsqueeze(0).to(image.device)

		return noised_image
