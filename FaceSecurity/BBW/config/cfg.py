# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -4.5
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01
fre_mul = 2

lamda_reconstruction = 1
lamda_guide = 2
lamda_low_frequency_secret = 0.3
lamda_discriminator = 8
lamda_discriminator_wm = 8

discriminator_lr = 0.001
template_lr = 0.0001

train_next = False
trained_epoch = 0

device_ids = [0]
device = "cuda:0"

template_size = 256

select_imgs_size = 32

MODEL_DESC = "train/"
MODEL_PATH_FRE = './results/' + MODEL_DESC

# Load:
suffix = 'model_checkpoint_00057.pt'
suffix_fre = 'HiNet_patchGAN_model_checkpoint_00057.pt'  # 11-》82  14-》 88 19-》91 23-》92.54
# suffix_fre = 'HiNet_patchGAN_model_checkpoint_00038.pt'  # 11-》82  14-》 88 19-》91 23-》92.54
suffix_org = 'model_checkpoint_00008.pt'

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

# Dataset
TRAIN_PATH = '/Data/FFHQ256/landmark/train/'  # 29781  SepMark->24183
VAL_PATH = '/Data/FFHQ256/image/val/'  # 3k
# TEST_PATH = '/Data/FFHQ256/landmark/test/'  # 3k

TEST_PATH = '/Data/FFHQ256/image/test/'  # 3k
# TEST_PATH = '/Data/CelebA-HQ/CelebA-HQ-256/test_256/'  # 3k
# TEST_PATH = '/Data/VGGFace2-256/'  # 3k

TARGET_PATH = "/Data/FFHQ256/image/target/"  # test 1k val 1k

format_train = 'npy'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False

# starGAN config
# style_dim = 64
# w_hpf = 1.0
# latent_dim = 16
# num_domains = 2
# num_workers = 1
# resume_iter = 100000
# checkpoint_dir = "/home/cw/ysc/HiNetlhb001/starganV2master/expr/checkpoints/celeba_hq"
# wing_path = '/home/cw/ysc/HiNetlhb001/starganV2master/expr/checkpoints/wing.ckpt'
# lm_path = '/home/cw/ysc/HiNetlhb001/starganV2master/expr/checkpoints/celeba_hq/celeba_lm_mean.npz'

# Saving checkpoints:
preview_upscale = 1.5

checkpoint_on_error = True
SAVE_freq = 1

# set
set_channels = 3
template_strength = 0.3

# 判别器数量12
discriminator_number = 2
discriminator_pred_dim = 1
discriminator_feature_dim = 2