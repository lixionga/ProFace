"""
This file defines the core research contribution
"""
import sys
sys.path.append('/data/lk/ID-disen4/models/attr_encoder')
sys.path.append("/data/lk/ID-disen4")
# import matplotlib
# matplotlib.use('Agg')
import math
from train_options import TrainOptions
import torch
from torch import nn
from models.attr_encoder import psp_encoders
from models.attr_encoder.StyleGan2.model import Generator
from configs.config_paths import model_paths

import numpy as np
import os
import json
import pprint
def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)

			self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

	def forward(self, x, latent_mask=None, input_code=False,
	            inject_latent=None, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0
		return codes

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None

if __name__ == '__main__':
	# from Configs import Global_Config
	opts = TrainOptions().parse()
	# if os.path.exists(opts.exp_dir):
	# 	raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	# os.makedirs(opts.exp_dir)
	#
	# opts_dict = vars(opts)
	# pprint.pprint(opts_dict)
	# with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
	# 	json.dump(opts_dict, f, indent=4, sort_keys=True)

	test_id = np.random.rand(8, 3, 256, 256)
	test_id = torch.from_numpy(test_id)
	test_id = test_id.to(torch.float32)
	test_id = test_id
	print(f'test_id dim len {len(test_id.shape)}')
	net = pSp(opts)
	with torch.no_grad():
		out = net(test_id)
	#
	# print(f'out type is {type(out)}')
	# print(f'out shape is{out.shape}')
	# test = out[0]
	# test_sum = torch.sum(test, dim=1)
	# print(f'out is {test_sum}')

	# id = np.random.rand(8, 1, 3)
	# attr = np.random.rand(8, 4, 3)
	# id = torch.from_numpy(id)
	# attr = torch.from_numpy(attr)
	# id = id.to(torch.float32)
	# attr = attr.to(torch.float32)
	# id = id.to(Global_Config.device)
	# attr = attr.to(Global_Config.device)
	# a, b = torch.broadcast_tensors(id, attr)
	# print(f'a type is {type(a)}')
	# print(f'a shape is{a.shape}')
	# print(f'a  is{a}')
	# print(f'b type is {type(b)}')
	# print(f'b shape is{b.shape}')
	# print(f'b  is{b}')