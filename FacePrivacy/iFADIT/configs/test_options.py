from argparse import ArgumentParser
import os
import sys
sys.path.append('/data/lk/ID-disen4')
from configs.config_paths import model_paths
class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--stylegan_size', default= 256, type=int, help="stylegan image size")
		self.parser.add_argument('--ir_se50_weights', default='/home/yl/lk/code/ID-dise4/model_ir_se50.pth', type=str, help="Path to facial recognition network used in ID loss")
		self.parser.add_argument('--id_cos_margin', default=0.1, type=float, help='margin for id cosine similarity')
		self.parser.add_argument('--mapper_weight', default="/home/yl/lk/code/RiDDLE-master/pretrained_models/iteration_90000.pt",type=str, help="The latents for the validation")
		self.parser.add_argument('--mapper_type', default='transformer', type=str, help="The latents for the validation")
		self.parser.add_argument('--transformer_normalize_type', default="layernorm", type=str, help="The latents for the validation")
		self.parser.add_argument('--transformer_split_list', nargs='+', type=int,default=[4,4,6])
		self.parser.add_argument('--transformer_add_linear', default=True,action="store_false")		
		self.parser.add_argument('--transformer_add_pos_embedding', default=True,action="store_false")		
		self.parser.add_argument('--latent_path', default="embeddings/invert_w_256.pt", type=str, help="The id embedding for the training")
		self.parser.add_argument('--image_path', default="/userHOME/yl/lk_data/ffhq256", type=str, help="The id embedding for the training")
		self.parser.add_argument('--exp_dir', default="experiments/exp_test", type=str, help="The id embedding for the testing")
		self.parser.add_argument('--image_size', default=256, type=int, help="The latents for the validation")
		self.parser.add_argument('--pwd_num', default=6, type=int, help="The latents for the validation")
		self.parser.add_argument('--batch_size', default=4, type=int, help="The latents for the validation")
		self.parser.add_argument('--latents_num', default=14, type=int, help="The latents for the validation")
		self.parser.add_argument('--morefc_num', default=0, type=int, help="The latents for the validation")
		self.parser.add_argument('--interpolate_batch_num', default=-1, type=int, help="The latents for the validation")
		self.parser.add_argument('--device', default="cuda:0", type=str, help="The id embedding for the testing")
		
		self.parser.add_argument('--id_encoder_path', default="/data/lk/ID-disen4/pretrained_models/model_ir_se50.pth", type=str, help="The id embedding for the testing")
		self.parser.add_argument('--attr_encoder_path', default="/data/lk/ID-disen4/pretrained_models/ckptattr_encoder_MKIHUPYWYPCR_2301.pt", type=str, help="The id embedding for the testing")
		self.parser.add_argument('--id_transformer_path', default="/data/lk/ID-disen4/pretrained_models/G_net_ffhq15INN607_20_202528.pth", type=str, help="The id embedding for the testing")
		self.parser.add_argument('--mlp_path', default="/data/lk/ID-disen4/pretrained_models/ckptmaper_MKIHUPYWYPCR_23010.pt", type=str, help="The id embedding for the testing")
		self.parser.add_argument('--icl_path', default="/data/lk/ID-disen4/pretrained_models/fuse_mlp_ffhq15INN607_20_202528.pth", type=str, help="The id embedding for the testing")
		self.parser.add_argument('--deghosting_path', default="/data/lk/ID-disen4/pretrained_models/deghosting_weight_1_140.pth", type=str, help="The id embedding for the testing")
		self.parser.add_argument('--data_name', default='Recovery&Disentangled_FFHQ*', type=str, help="The id embedding for the testing")
		self.parser.add_argument('--use_one_batch', default=True, type=str, help="The id embedding for the testing")

		self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
		self.parser.add_argument('--output_size', default=256, type=int, help='Output size of generator')

		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
		self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--start_from_latent_avg', default=False, help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', default=False, help='Whether to learn in w space instead of w+')

		self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
		self.parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
		self.parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')
		self.parser.add_argument('--moco_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')

		self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

		# arguments for weights & biases support
		self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None, help='For super-res, comma-separated resize factors to use for inference.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts