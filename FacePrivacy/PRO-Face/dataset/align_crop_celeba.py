# Lin Yuan scripts, referred from https://github.com/LynnHo/HD-CelebA-Cropper
import os
import face_align_ffhqandnewarc as face_align
import cv2
import numpy as np
from multiprocessing import Pool
import shutil
import tqdm


n_worker = 8
crop_size = 224
img_dir = '/mnt/Data8T/Datasets/CelebA/img_celeba'
img_dir_aligned = '/mnt/Data8T/Datasets/CelebA/align_crop_224'
img_dir_targets = '/mnt/Data8T/Datasets/CelebA/align_crop_224/valid_frontal/all'
lmk_file = '/mnt/Data8T/Datasets/CelebA/Anno/list_landmarks_celeba.txt'

if not os.path.isdir(img_dir_aligned):
    os.makedirs(img_dir_aligned)
if not os.path.isdir(img_dir_targets):
    os.makedirs(img_dir_targets)

n_landmark = 5
img_names = np.genfromtxt(lmk_file, skip_header=2, dtype=np.str, usecols=0)
landmarks = np.genfromtxt(lmk_file, skip_header=2, dtype=np.float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1, n_landmark, 2)

# Test id from 182638
def work(i):  # a single work
    for _ in range(3):  # try three times
        try:
            img_name = img_names[i]
            M, index_M = face_align.estimate_norm(landmarks[i], crop_size, mode=None)
            img = cv2.imread(os.path.join(img_dir, img_name))
            img_aligned = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

            path_aligned = os.path.join(img_dir_aligned, img_name)
            cv2.imwrite(path_aligned, img_aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # Copy those test image that has frontal face (index_M = 2) into another folder as target images for swap
            if i >= 182637 and index_M == 2:
                shutil.copy(path_aligned, img_dir_targets)

            name_landmark_str = ('%s' + ' %.1f' * 6) % ((img_name, ) + tuple(M.flatten()))
            succeed = True
            break
        except:
            succeed = False
    if succeed:
        return name_landmark_str
    else:
        print('%s fails!' % img_names[i])


pool = Pool(n_worker)
# name_landmark_strs = list(tqdm.tqdm(pool.imap(work, range(len(img_names))), total=len(img_names)))
image_range = range(182637)
name_landmark_strs = list(tqdm.tqdm(pool.imap(work, image_range), total=len(image_range)))
pool.close()
pool.join()


print("Done")
