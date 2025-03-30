import os
import torch
DIR_HOME = os.path.expanduser("~")
DIR_DATASET = os.path.join(DIR_HOME, 'Datasets')
DIR_PROJECT = os.path.join(DIR_HOME, 'project/ProFaceInv')
GPU0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
GPU1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# This is the device that SimSwap relies on: 
# DEVICE = GPU0
DEVICE = GPU0   # 1卡训练结束后修改
# WRONG_RECOVER_TYPE = 'Random'
WRONG_RECOVER_TYPE = 'Obfs'



INV_BLOCKS = 3

SECRET_KEY_AS_NOISE = True  # edit to False after later
ADJ_UTILITY = False

class Utility():
    NONE = 0
    FACE = 1
    GENDER = 2
    IDENTITY = 3

utility_level = Utility.NONE

utility_weights = {
    Utility.FACE: 20.0,
    Utility.GENDER: 2.0,
    Utility.IDENTITY: 5.0,
}


debug = False
batch_size = 6  # before:6



# Image and face_detection save period
SAVE_IMAGE_INTERVAL = 1000
SAVE_MODEL_INTERVAL = 5000

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


# Pretrained model
DIR_FACESHIFTER_G_LATEST = os.path.join(DIR_PROJECT, 'FaceShifter/saved_models/G_latest.pth')
DIR_FACESHIFTER_IRSE50 = os.path.join(DIR_PROJECT, 'FaceShifter/face_modules/model_ir_se50.pth')

DIR_SIMSWAP_NET_G = os.path.join(DIR_PROJECT, 'SimSwap/models/checkpoints/people/latest_net_G.pth')
DIR_ARC_PATH = os.path.join(DIR_PROJECT, 'SimSwap/arcface_model/arcface_checkpoint.tar')

# for test
# DIR_INN = os.path.join(DIR_PROJECT, 'experiments/checkpoints_256/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth')
# DIR_INN = '/home/KYO/project/ProFaceInv/runs/Jan08_23-38-58_lhb003_simswap_inv3_recTypeObfs_secretAsNoise_TripMargin1.2/checkpoints/simswap_inv3_recTypeObfs_secretAsNoise_TripMargin1.2_ep2.pth'  # 只模拟
DIR_INN = '/home/KYO/project/ProFaceInv/runs/Jan15_01-23-23_lhb003_simswap_inv3_recTypeObfs_secretAsNoise_TripMargin1.2/checkpoints/simswap_inv3_recTypeObfs_secretAsNoise_TripMargin1.2_ep6_iter15000.pth'  # 模拟+补偿
# DIR_INN = os.path.join(DIR_PROJECT, '/home/KYO/project/ProFaceInv/runs/Jan02_13-56-03_lhb003_simswap_inv3_recTypeRandom/checkpoints/simswap_inv3_recTypeRandom_ep5_iter5000.pth')  # mlp1
# DIR_INN = os.path.join(DIR_PROJECT, 'runs/Jan10_21-32-31_lhb003_simswap_inv3_recTypeObfs_secretAsNoise_TripMargin1.2/checkpoints/simswap_inv3_recTypeObfs_secretAsNoise_TripMargin1.2_ep18_iter5000.pth')  # 初稿

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


latest_G_dir = os.path.join(DIR_PROJECT, 'SimSwap/models/checkpoints')

dataset_dir = os.path.join(DIR_DATASET, 'CelebA/align_crop_224')

target_img_dir_train = os.path.join(DIR_DATASET, 'CelebA/align_crop_224/train')
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

if __name__ == '__main__':
    import torch

    print(DIR_INN)  # 查看 PyTorch 版本
