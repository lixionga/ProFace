import os
import torch
DIR_HOME = os.path.expanduser("~")
DIR_DATASET = os.path.join(DIR_HOME, 'Datasets')
DIR_PROJECT = os.path.join(DIR_HOME, 'ProFace/FacePrivacy/PRO-Face U')
GPU0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
GPU1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

DEVICE = GPU0
# WRONG_RECOVER_TYPE = 'Obfs'
# DEVICE = GPU1
WRONG_RECOVER_TYPE = 'Random'

INN_checkpoints="simswap_inv3_recTypeRandom_utility_ep23.pth"
FC_checkpoints="simswap_inv3_recTypeRandom_utility_utilityFC_ep23.pth"
mlp_checkpoints="simswap_inv3_recTypeRandom_utility_ep3_iter500.pth"
INV_BLOCKS = 3

SECRET_KEY_AS_NOISE = True
ADJ_UTILITY = True

class Utility():
    NONE = 0
    FACE = 1
    GENDER = 2
    IDENTITY = 3

utility_level = Utility.GENDER

utility_weights = {
    Utility.FACE: 20.0,
    Utility.GENDER: 2.0,
    Utility.IDENTITY: 5.0,
}


debug = False
batch_size = 16



# Image and face_detection save period
SAVE_IMAGE_INTERVAL = 200
SAVE_MODEL_INTERVAL = 3000

# Training parameters
# recognizer = 'InceptionResNet'
# recognizer = 'IResNet100'
recognizer = 'AdaFaceIR100'


# obfuscator = 'medianblur_15'
# obfuscator = 'blur_21_6_10' #'blur_21_2_6'
# obfuscator = 'pixelate_9' #'pixelate_4_10'
# obfuscator = 'faceshifter'
obfuscator = 'simswap'
# obfuscator = 'mask'
# obfuscator = 'hybrid'
# obfuscator = 'hybridMorph'
# obfuscator = 'hybridAll'

# Training weights for different face recognizers and obfuscations.
# The three numbers indicate the weights of loss_triplet_p2p, loss_triplet_p2o and loss_utility respectively.
identity_weights = {
    'InceptionResNet':
        {
            'medianblur': (1.0, 0.2), #(5, 1), OK
            'blur': (1.0, 0.3), #(5, 1), ?????
            'pixelate': (1.0, 0.25), #(5, 1), OK
            'faceshifter': (0.1, 0.1),
            'simswap': (0.075, 0.075),
            'hybrid': (1.0, 0.2),
         }, #OK
    'IResNet100':
        {
            'medianblur': (0.1, 0.1), #OK
            'blur': (0.1, 0.1), #OK
            'pixelate': (0.1, 0.1), #OK
            'faceshifter': (0.1, 0.1),
            'simswap': (0.0075, 0.02),
            'hybrid': (0.1, 0.1),
        }, #OK
    'AdaFaceIR100':
        {
            'medianblur': (1.0, 0.2), #(5, 1), #OK
            'blur': (1.0, 0.2), #(5, 1), #OK
            'pixelate': (1.0, 0.2), #OK
            'faceshifter': (0.1, 0.1),
            'simswap': (0.1, 0.1), #OK
            'hybrid': (1.0, 0.2),
            'hybridMorph': (0.1, 0.1),
            'hybridAll': (0.1, 0.1),
        },
}


gender_weight = 0.1


dataset_dir = os.path.join(DIR_DATASET, 'CelebA/align_crop_224')

target_img_dir_train = os.path.join(DIR_DATASET, 'CelebA/align_crop_224/valid_frontal')
target_img_dir_test = os.path.join(DIR_DATASET, 'CelebA/align_crop_224/test_frontal')
celeba_attr_file = os.path.join(DIR_DATASET, 'CelebA/Anno/list_attr_celeba.txt')
eval_dir = os.path.join(DIR_DATASET, 'LFW/LFW_112_test_pairs')
eval_pairs = os.path.join(DIR_DATASET, 'LFW/pairs.txt')
lfw_male_file = os.path.join(DIR_DATASET, 'LFW/male_names.txt')
lfw_female_file = os.path.join(DIR_DATASET, 'LFW/female_names.txt')
mini_imagenet_dir = os.path.join(DIR_DATASET, 'mini-imagenet')
# lfw_male_file = '/Users/yuanlin/Datasets/LFW/LFW male-female files/male_names.txt'
# lfw_female_file = '/Users/yuanlin/Datasets/LFW/LFW male-female files/female_names.txt'
attr_rec_model = f'face/gender_model/gender_classifier_{recognizer}.pth'
cartoon_face_path = os.path.join(DIR_DATASET, 'CartoonSet')


# Path to saved checkpoints
MODEL_PATH = [
    # os.path.join(PROJECT_DIR, 'checkpoints/pixelate_4_10_IResNet100_ep3_iter16600.pth'),
    # os.path.join(PROJECT_DIR, 'checkpoints/pixelate_4_10_IResNet100_ep3_iter7000.pth'),
    # os.path.join(PROJECT_DIR, 'checkpoints/pixelate_4_10_IResNet100_ep2_iter9000.pth'),
    # os.path.join(PROJECT_DIR, 'checkpoints/pixelate_4_10_IResNet100_ep1_iter3000.pth')
]

# Super parameters
clamp = 2.0
channels_in = 3
# log10_lr = -3.5 #-4.5
# lr = 10 ** log10_lr
# lr = 0.000125 # for INV_block
lr = 0.00001 # for INV_block_affine
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0, 1]

# Train:

cropsize = 224
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 1024
batchsize_val = 2
shuffle_val = False
val_freq = 50


# Dataset
TRAIN_PATH = '/home/jjp/Dataset/DIV2K/DIV2K_train_HR/'
VAL_PATH = '/home/jjp/Dataset/DIV2K/DIV2K_valid_HR/'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

# MODEL_PATH = '/home/jjp/Hinet/face_detection/'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = '/home/jjp/Hinet/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'

# Load:
suffix = 'face_detection.pt'
tain_next = False
trained_epoch = 0
