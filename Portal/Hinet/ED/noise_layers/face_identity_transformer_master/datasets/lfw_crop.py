import os.path as osp
import numpy as np
from PIL import Image

import torch.utils.data as data
import torch

__all__ = ['LFW_CROP']

EXTENSION_FACTOR = 2


class LFW_CROP(data.Dataset):
    def __init__(self, train, transform, args):
        self.root = osp.join(args.data_root, 'lfw')
        self.transform = transform

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

        self.anno_dict = {anno[0]: anno for anno in self.raw_annotations}

        bbox_path = osp.join(args.data_root, 'lfw_detection.txt')
        self.bbox_dict = {}
        with open(bbox_path) as fd:
            bbox_lines = [bbox_line.strip().split() for bbox_line in fd.readlines()]

        for bbox_line in bbox_lines:
            if bbox_line[0] not in self.anno_dict:
                continue
            oleft = float(bbox_line[1])
            oup = float(bbox_line[2])
            oright = float(bbox_line[3])
            odown = float(bbox_line[4])

            width = oright - oleft
            new_width = width * EXTENSION_FACTOR
            x_margin = (new_width - width) / 2
            y_margin = (new_width - (odown - oup)) / 2  # MAY BE NEED CHANGE

            box_left = max(int(oleft - x_margin), 0)
            box_right = min(int(oright + x_margin), 249)
            box_up = max(int(oup - y_margin), 0)
            box_down = min(int(odown + y_margin), 249)

            new_width = box_right - box_left
            new_height = box_down - box_up

            for i in range(5):
                self.anno_dict[bbox_line[0]][2 * i + 1] = (self.anno_dict[bbox_line[0]][
                                                               2 * i + 1] - box_left) / new_width * 250.
                self.anno_dict[bbox_line[0]][2 * i + 2] = (self.anno_dict[bbox_line[0]][
                                                               2 * i + 2] - box_up) / new_height * 250.

            self.bbox_dict[bbox_line[0]] = [box_left,
                                            box_up,
                                            box_right,
                                            box_down]
            # extended left, right, up, down

    def __len__(self):
        return len(self.raw_annotations)

    def __getitem__(self, index):
        anno = self.anno_dict[self.raw_annotations[index][0]]

        img_path = osp.join(self.root, anno[0])
        label = 0
        landmarks = torch.empty((5, 2), dtype=torch.float32)
        for i in range(5):
            landmarks[i, 0] = anno[2 * i + 1]
            landmarks[i, 1] = anno[2 * i + 2]

        img = Image.open(img_path).convert("RGB")
        bbox = self.bbox_dict[anno[0]]
        img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        if self.transform is not None:
            img = self.transform(img)

        return img, label, landmarks, img_path

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of imgs: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__str__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str
