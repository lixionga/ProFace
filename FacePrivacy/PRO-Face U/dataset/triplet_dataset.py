from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def default_image_loader(path):
    return Image.open(path)


def get_celeba_gender_labels(attr_file):
    img_gender_dict = dict(np.genfromtxt(attr_file, skip_header=2, dtype=str, usecols=[0, 21]))
    new_dict = {k: 1 if img_gender_dict[k] == '1' else 0 for k in img_gender_dict.keys()}
    return new_dict


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, filenames_filename, triplets_file_name, transform=transforms.ToTensor(),
                 attr_file='', loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers,
                where integer i refers to the i-th image in the filenames file.
                For a line of intergers 'a b c', a triplet is defined such that image a is more
                similar to image c than it is to image b, e.g.,
                0 2017 42 """
        self.base_path = base_path
        self.filenamelist = []
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        triplets = []
        for line in open(triplets_file_name):
            triplets.append((line.split()[0], line.split()[1], line.split()[2])) # anchor, far, close
        self.triplets = triplets
        self.transform = transform
        self.loader = loader
        self.celeba_gender_dict = {}
        if attr_file and os.path.isfile(attr_file):
            self.celeba_gender_dict = get_celeba_gender_labels(attr_file=attr_file)

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets[index]
        img1_name = self.filenamelist[int(path1)]
        img1_gender = self.celeba_gender_dict[os.path.basename(img1_name)]
        img1 = self.loader(os.path.join(self.base_path, img1_name))
        img2_name = self.filenamelist[int(path2)]
        img2_gender = self.celeba_gender_dict[os.path.basename(img2_name)]
        img2 = self.loader(os.path.join(self.base_path, img2_name))
        img3_name = self.filenamelist[int(path3)]
        img3_gender = self.celeba_gender_dict[os.path.basename(img3_name)]
        img3 = self.loader(os.path.join(self.base_path, img3_name))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img1_name, img1_gender), \
               (img2, img2_name, img2_gender), \
               (img3, img3_name, img3_gender)

    def __len__(self):
        return len(self.triplets)


def save_list_to_file(lst, filepath):
    with open(filepath, mode='wt', encoding='utf-8') as f:
        f.write('\n'.join(lst))


def generate_triplet_text_for_celeba(root_celeba='/home/yuanlin/Datasets/CelebA/img_align_celeba_crop_160'):
    import random
    from tqdm import tqdm
    for split in ('valid', 'test', 'train'):
        celeba_subset_dir = f'{root_celeba}/{split}'
        file_filename_list = f'{root_celeba}/filenames_{split}.txt'
        file_triplet_list = f'{root_celeba}/triplets_{split}.txt'

        list_filenames = []
        # Set the directory you want to start from
        for dirname, subdir_list, file_list in os.walk(celeba_subset_dir):
            person_id = os.path.basename(dirname)
            for fname in file_list:
                image_short_path = os.path.join(person_id, fname)
                list_filenames.append(image_short_path)
        print(f'* {split}: {len(list_filenames)} images')


        # For every image file, treated as anchor, find a pair of positive and negative samples
        list_triplet = []
        for idx, filename in enumerate(tqdm(list_filenames)):
            id, _ = filename.split('/')
            positive_candidates = list(filter(lambda f: f.startswith(id) and f != filename, list_filenames))
            negative_candidates = list(filter(lambda f: not f.startswith(id), list_filenames))
            a = filename
            p = random.choice(positive_candidates) if positive_candidates else a
            n = random.choice(negative_candidates)
            p_idx = list_filenames.index(p)
            n_idx = list_filenames.index(n)
            list_triplet.append(f'{idx} {n_idx} {p_idx}')

        save_list_to_file(list_filenames, file_filename_list)
        save_list_to_file(list_triplet, file_triplet_list)
        print("Finish writing {} and {}".format(file_filename_list, file_triplet_list))


if __name__ == '__main__':
    # generate_triplet_text_for_celeba()
    root_celeba = '/home/yuanlin/Datasets/CelebA/align_crop_224'
    celeba_subset_dir = f'{root_celeba}/test'
    file_filename_list = f'{root_celeba}/filenames_test.txt'
    file_triplet_list = f'{root_celeba}/triplets_test.txt'
    triplet_dataset = TripletDataset(celeba_subset_dir, file_filename_list, file_triplet_list)
    loader = DataLoader(triplet_dataset, num_workers=8, batch_size=8, shuffle=False)

    print('Done')
