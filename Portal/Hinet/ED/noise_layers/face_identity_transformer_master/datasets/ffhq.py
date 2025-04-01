import glob
import os.path as osp
from PIL import Image
import torch.utils.data as data
import torch

__all__ = ['FFHQ']


class FFHQ(data.Dataset):
    def __init__(self, train, transform, args):
        self.root = args.data_root
        self.transform = transform

        # landmark_path = osp.join(self.root, 'validation_paths.txt')

        # with open(landmark_path) as fd:
        #     self.raw_annotations = [line.strip() for line in fd.readlines()]

        self.files = sorted(glob.glob(self.root + "/*.png"))

    def __len__(self):
        # return len(self.raw_annotations)
        return len(self.files)

    def __getitem__(self, index):
        # anno = self.raw_annotations[index]

        image_path = self.files[index]

        img = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, image_path

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of imgs: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str
