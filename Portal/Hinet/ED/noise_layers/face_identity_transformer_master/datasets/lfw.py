# For test only
# 5749 people, 13233 imgs, all in shape (250, 250, 3), uint8

import os.path as osp
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch

__all__ = ['LFW']


class LFW(data.Dataset):
    def __init__(self, train, transform, args):
        landmark_path = osp.join(args.data_root, 'lfw_landmark.txt')

        with open(landmark_path) as fd:
            self.raw_annotations = [line.strip().split() for line in fd.readlines()]

        for idx in range(len(self.raw_annotations)):
            self.raw_annotations[idx] = self.raw_annotations[idx][0:1] + [
                float(item) for item in self.raw_annotations[idx][1:]]

        if not args.evaluate:
            test_id_indices = set(np.random.choice(len(self.raw_annotations), size=args.test_size, replace=False))
            self.raw_annotations = [anno for idx, anno in enumerate(self.raw_annotations) if
                                    idx in test_id_indices]

        self.root = osp.join(args.data_root, 'lfw')
        self.transform = transform

    def __len__(self):
        return len(self.raw_annotations)

    def __getitem__(self, index):
        anno = self.raw_annotations[index]

        img_path = osp.join(self.root, anno[0])
        label = 0
        landmarks = torch.empty((5, 2), dtype=torch.float32)
        for i in range(5):
            landmarks[i, 0] = anno[2 * i + 1]
            landmarks[i, 1] = anno[2 * i + 2]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label, landmarks, img_path

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of imgs: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str
