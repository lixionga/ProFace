import os
import torch
DIR_HOME = os.path.expanduser("~")

# Set the correct path of this project
PROJECT_DIR = os.path.join(DIR_HOME, 'Projects/PRO-Face')

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Training parameters
# Select face recognizer from ['MobileFaceNet', 'InceptionResNet', 'IResNet50', 'IResNet100', 'SEResNet50']
recognizer = 'SEResNet50'

# Select obfuscator from ['blur_31_2_8', 'pixelate_7', 'faceshifter', 'simswap']
obfuscator = 'blur_31_2_8'

# Path to training dataset
dataset_dir = os.path.join(DIR_HOME, 'Datasets/CelebA/align_crop_224')

# Path to target images (for face swap only) in training
target_img_dir_train = os.path.join(DIR_HOME, 'Datasets/CelebA/align_crop_224/valid_frontal')

# Path to target images (for face swap only) in test
target_img_dir_test = os.path.join(DIR_HOME, 'Datasets/CelebA/align_crop_224/test_frontal')

# Run training in debug mode
debug = False

# Image and model save period
SAVE_IMAGE_INTERVAL = 1000
SAVE_MODEL_INTERVAL = 5000

# Batch size of training
batch_size = 8
