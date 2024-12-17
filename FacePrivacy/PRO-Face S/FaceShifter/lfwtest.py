import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import os
from torchvision.utils import save_image

from PIL import Image
from face_shifter import face_shifter_batch


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


import config.config as c
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = c.DEVICE


def main():
    data_dir = '/mnt/Data8T/lyyang/LFW_align_crop_224'
    targ_img_path = '/home/lyyang/papercode/FRModel/testimg/Ethan_Hawke_0001.jpg'
    batch_size = 8
    workers = 0 if os.name == 'nt' else 8

    trans = transforms.Compose([
        # transforms.Resize(160),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    target_image = trans(Image.open(targ_img_path))
    target_img = target_image.repeat(batch_size, 1, 1, 1)

    dataset = datasets.ImageFolder(data_dir, transform=trans)

    dataset.samples = [
        (p, (p, idx))
        for p, idx in dataset.samples
    ]

    test_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )

    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(test_loader):
            batch_idx += 1
            source_img, (paths, yb) = batch
            source_img = source_img.to(device)
            img_path_orig = f"/home/lyyang/papercode/FRModel/testimg/lfw/{batch_idx}_org.jpg"
            img_path_res = f"/home/lyyang/papercode/FRModel/testimg/lfw/{batch_idx}_res.jpg"
            save_image((source_img + 1.0) / 2.0, img_path_orig, nrow=4)
            face_shifter_tensor = face_shifter_batch(source_img, target_img)
            save_image((face_shifter_tensor + 1.0) / 2.0, img_path_res, nrow=4)



if __name__ == '__main__':
    # 修改之前的target
    import time

    T1 = time.time()
    main()
    T2 = time.time()
    with open("/home/lyyang/papercode/FRModel/testimg/timelog.txt", "w") as f:
        f.write('程序运行时间:%s秒' % (T2 - T1))
