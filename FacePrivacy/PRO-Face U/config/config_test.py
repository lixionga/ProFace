import os
import torch
DIR_HOME = os.path.expanduser("~")
DIR_DATASET = os.path.join(DIR_HOME, 'Datasets')
DIR_PROJECT = os.path.join(DIR_HOME, 'ProFace/FacePrivacy/PRO-Face U')


GPU0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
GPU1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
DEVICE = GPU1

# Path to original datasets
datasets = {
    'CelebA': os.path.join(DIR_DATASET, 'CelebA/align_crop_224/test'),
    'LFW': os.path.join(DIR_DATASET, 'LFW/LFW_align_crop_224_test_pairs'),
    'VGGFace2': os.path.join(DIR_DATASET, 'VGG-Face2/data/test_align_crop_224'),
}

# Path to original datasets
datasets1k = {
    'CelebA': os.path.join(DIR_PROJECT, 'experiments/test_data/CelebA'),
    'LFW': os.path.join(DIR_PROJECT, 'experiments/test_data/LFW'),
    'VGGFace2': os.path.join(DIR_PROJECT, 'experiments/test_data/VGGFace2'),
}

target_image_dataset_dir = os.path.join(DIR_DATASET, 'CelebA/align_crop_224/test_frontal')
cartoon_face_path = os.path.join(DIR_DATASET, 'CartoonSet')
