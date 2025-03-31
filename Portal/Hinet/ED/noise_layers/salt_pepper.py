import torch
import torch.nn as nn
import numpy as np


class SaltPepper(nn.Module):

	def __init__(self, prob=0.1):
		super(SaltPepper, self).__init__()
		self.prob = prob

	def sp_noise(self, image, prob):
		mask = torch.Tensor(np.random.choice((0, 1, 2), image.shape[2:], p=[1 - prob, prob / 2., prob / 2.])).to(image.device)
		mask = mask.expand_as(image)

		image[mask == 1] = 1  # salt
		image[mask == 2] = -1  # pepper

		return image

	def forward(self, image_cover_mask):
		image, mask = image_cover_mask[0], image_cover_mask[2]

		#mask = mask[:, 0: 3, :, :]
		return self.sp_noise(image, self.prob) #image * mask + self.sp_noise(image, self.prob) * (1 - mask)

