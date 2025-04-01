import torch
import torch.nn as nn
import numpy as np


class FaceCrop(nn.Module):

	def __init__(self, prob=0.035):
		super(FaceCrop, self).__init__()
		self.height_ratio = int(np.sqrt(prob) * 100) / 100
		self.width_ratio = int(np.sqrt(prob) * 100) / 100

	def forward(self, image_cover_mask):
		image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]

		mask = mask[:, 0: 3, :, :]

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		maskk = torch.zeros_like(image)
		maskk[:, :, h_start: h_end, w_start: w_end] = 1

		return image * mask #image * mask + image * maskk * (1 - mask)


class FaceCropout(nn.Module):

	def __init__(self, prob=0.3):
		super(FaceCropout, self).__init__()
		self.height_ratio = int(np.sqrt(prob) * 100) / 100
		self.width_ratio = int(np.sqrt(prob) * 100) / 100

	def forward(self, image_cover_mask):
		image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]

		mask = mask[:, 0: 3, :, :]

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		output = cover_image.clone()
		output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
		return image * mask + cover_image * (1 - mask) #image * mask + output * (1 - mask)


class Dropout(nn.Module):

	def __init__(self, prob=0.3):
		super(Dropout, self).__init__()
		self.prob = prob

	def forward(self, image_cover_mask):
		image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]

		#mask = mask[:, 0: 3, :, :]

		maskk = torch.Tensor(np.random.choice([0.0, 1.0], image.shape[2:], p=[self.prob, 1 - self.prob])).to(image.device)
		maskk = maskk.expand_as(image)
		output = image * maskk + cover_image * (1 - maskk)
		return output #output * mask + image * (1 - mask)


class FaceErase(nn.Module):

	def __init__(self):
		super(FaceErase, self).__init__()

	def forward(self, image_cover_mask):
		image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]

		mask = mask[:, 0: 3, :, :]

		return image * (1 - mask)


class FaceEraseout(nn.Module):

	def __init__(self):
		super(FaceEraseout, self).__init__()

	def forward(self, image_cover_mask):
		image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]

		mask = mask[:, 3: 6, :, :]

		output = image * (1 - mask) + cover_image * mask
		return output


def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
	image_height = image_shape[2]
	image_width = image_shape[3]

	remaining_height = int(height_ratio * image_height)
	remaining_width = int(width_ratio * image_width)

	if remaining_height == image_height:
		height_start = 0
	else:
		height_start = np.random.randint(0, image_height - remaining_height)

	if remaining_width == image_width:
		width_start = 0
	else:
		width_start = np.random.randint(0, image_width - remaining_width)

	return height_start, height_start + remaining_height, width_start, width_start + remaining_width

