import torch
import torch.nn as nn


class Identity(nn.Module):
	"""
	Identity-mapping noise layer. Does not change the image
	"""

	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, image_cover_mask):
		image = image_cover_mask[0]
		return image
