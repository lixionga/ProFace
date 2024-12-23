#### Face detection and recognition training pipeline
from face_detection.mtcnn import MTCNN
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y


def main(data_dir):
    #### Define run parameters
    # The dataset should follow the VGGFace2/ImageNet-style directory layout. Modify `data_dir` to the location of the
    # dataset on wish to finetune on.
    output_dir = data_dir + '_crop_224'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    batch_size = 128
    workers = 0 if os.name == 'nt' else 8

    #### Determine if an nvidia GPU is available
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Running on device: {}'.format(device))

    #### Define MTCNN module
    mtcnn = MTCNN(image_size=224, margin=0, device=device)

    print('Dataset preprocessing')
    #### Perfom MTCNN facial detection: Iterate through the DataLoader object and obtain cropped faces.
    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(data_dir, output_dir))
        for p, _ in dataset.samples
    ]

    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=collate_pil
    )

    print('Starting detection')
    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

    # Remove mtcnn to reduce GPU memory usage
    del mtcnn


if __name__ == "__main__":
    main('/home/yuanlin/Datasets/VGG-Face2/data/train')
