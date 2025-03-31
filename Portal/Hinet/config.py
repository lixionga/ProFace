# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -4.5
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5  # L2正则化，防止过拟合
init_scale = 0.01
fre_mul = 2

lamda_reconstruction = 1
lamda_guide = 2
lamda_low_frequency = 1
lamda_low_frequency_secret = 0.3
lamda_discriminator = 8

discriminator_lr = 0.001
template_lr = 0.0001

tain_next = False
trained_epoch = 2

MODEL_DESCRIPTION = "FFHQ256-inv4-3fun-vector-1-2-1-0.3-8-seed25-resize0.5/"
MODEL_PATH_FRE = '/home/ysc/HiNet/model/final/' + MODEL_DESCRIPTION
suffix_fre = 'HiNet_patchGAN_model_checkpoint_00057.pt'  # 57 ->best

device_ids = [1]
device = "cuda:1"

# Dataset
# FFHQ original 70k
TRAIN_PATH = '/Data/FFHQ256/landmark/train/'  # 29781 png
VAL_PATH = '/Data/FFHQ256/image/val/'  # 3000

TEST_PATH = '/Data/FFHQ256/image/test/'  # 3000 png
# TEST_PATH = '/Data/CelebA-HQ/CelebA-HQ-256/HiNet/image/test/'  # 3000 jpg
# TEST_PATH = '/Data/vggface2-224/test_256/'  # 3000 jpg

TARGET_PATH = "/Data/FFHQ256/image/target/"  # 2k val 1k test 1k

# CelebA HQ
# TRAIN_PATH = '/Data/CelebA-HQ/CelebA-HQ-256/HiNet/landmark/train/'
# TEST_PATH = '/Data/CelebA-HQ/CelebA-HQ-256/HiNet/image/test/'  # 3000 jpg
# TEST_PATH = '/Data/vggface2-224/test_256/'  # 3000 jpg
# VAL_PATH = '/Data/CelebA-HQ/CelebA-HQ-256/HiNet/landmark/val/'  # 3000
# TARGET_PATH = "/Data/CelebA-HQ/CelebA-HQ-256/HiNet/image/target/"  # 2k train 1k val 1k

StarGAN_DATA_PAHT = "/home/ysc/HiNet/starganV2master/data/celeba_hq/val/"
GANimation_DATA_PAHT = "/Data/FFHQ256/image/CelebHQ_GANimation_256/"

WID_DATA_PAHT = "/home/ysc/Watermarking_In_FaceID/Image/watermarked_CelebAHQ_CROP_256/sw(0.1)_st(gold)_arcface/reconstructed/"

format_landmark = 'npy'
format_img = 'png'
# format_img = 'jpg'

# Train:
batch_size = 4
cropsize = 256
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 256
batchsize_val = 1
shuffle_val = False
val_freq = 1

# batchsize_test = 8


# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False

# starGANv2 config
style_dim = 64
w_hpf = 1.0
latent_dim = 16
num_domains = 2
num_workers = 1
resume_iter = 100000
checkpoint_dir = "/home/ysc/HiNet/starganV2master/expr/checkpoints/celeba_hq"
wing_path = '/home/ysc/HiNet/starganV2master/expr/checkpoints/wing.ckpt'
lm_path = '/home/ysc/HiNet/starganV2master/expr/checkpoints/celeba_hq/celeba_lm_mean.npz'

# Saving checkpoints:
preview_upscale = 1.5

checkpoint_on_error = True
SAVE_freq = 1

IMAGE_PATH = '/home/ysc/HiNet/image/img_savaeall/mask/'
IMAGE_PATH_COVER = IMAGE_PATH + 'cover/'
IMAGE_PATH_SECRET = IMAGE_PATH + 'secret/'
IMAGE_PATH_STEG = IMAGE_PATH + 'steg/'
IMAGE_PATH_STEG_NM = IMAGE_PATH + 'steg_nm/'
IMAGE_PATH_SWAP_TARGET = IMAGE_PATH + 'target/'
IMAGE_PATH_SWAP_FAKE = IMAGE_PATH + 'steg_big_mask/'
IMAGE_PATH_SHOW_ALL = IMAGE_PATH + 'show_all/'
IMAGE_PATH_SECRET_REV = IMAGE_PATH + 'secret-rev/'
IMAGE_PATH_SECRET_MP_REV = IMAGE_PATH + 'secret-mp-rev-big/'

IMAGE_PATH_Diff_SECRET_NM = IMAGE_PATH + 'diff_secret_nm/'
IMAGE_PATH_Diff_SECRET_MP = IMAGE_PATH + 'diff_secret_mp/'

# set
set_channels = 3
template_size = 256
template_strength = 1

# 判别器数量12
discriminator_number = 2
discriminator_pred_dim = 1
discriminator_feature_dim = 2

DE_lr = 0.0001
DE_beta1 = 0.5
message_range = 0.1
message_length = 128
attention_encoder = "se"
lambda_decoder = 10
lambda_encoder = 1
