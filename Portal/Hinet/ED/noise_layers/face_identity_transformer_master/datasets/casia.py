import os.path as osp
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

__all__ = ['CASIA']


class CASIA(data.Dataset):

    def __init__(self, train, transform, args):
        landmark_path = osp.join(args.data_root, 'casia_landmark.txt')
        with open(landmark_path) as fd:
            self.raw_annotations = [line.strip().split() for line in fd.readlines()]

        for idx in range(len(self.raw_annotations)):
            self.raw_annotations[idx] = self.raw_annotations[idx][0:1] + [int(self.raw_annotations[idx][1])] + [
                float(item) for item in self.raw_annotations[idx][2:]]
        # path, id_label, 5 landmarks

        self.root = osp.join('/home/KYO/dataset', 'CASIA-WebFace')

        self.transform = transform
        self.train = train
        self.evaluate = args.evaluate
        self.val_during_training = hasattr(args, "during_training") and args.during_training

        all_identities = set([anno[1] for anno in self.raw_annotations])
        min_test_id = max(all_identities) // 5 * 4
        test_end_id = (len(all_identities) - min_test_id) // 2 + min_test_id
        # [0, min_test_id) : training; [min_test_id, test_end_id): testing; [test_end_id, end]: validation
        print('# all identities', len(all_identities),
              '# test ids', test_end_id - min_test_id,
              '# val ids', len(all_identities) - test_end_id)

        if self.train:  # train
            self.raw_annotations = [anno for anno in self.raw_annotations if anno[1] < min_test_id]
        elif args.evaluate:  # test on full test set
            self.raw_annotations = [anno for anno in self.raw_annotations if
                                    anno[1] >= min_test_id and anno[1] < test_end_id]
            print('evaluation # imgs', len(self.raw_annotations))
        elif hasattr(args, 'validation') and args.validation:  # validate on full val set
            self.raw_annotations = [anno for anno in self.raw_annotations if anno[1] >= test_end_id]
        elif hasattr(args, 'test_from_path') and args.test_from_path:
            self.raw_annotations = [anno for anno in self.raw_annotations if anno[0] in args.test_from_path]
        else:  # val on val / test on args.test_size imgs
            # during training, only use validation set
            if hasattr(args, "during_training") and args.during_training:
                tmp_raw_annotations = [anno for anno in self.raw_annotations if
                                       anno[1] >= test_end_id]
                print('visu val # imgs', args.test_size)
            else:
                tmp_raw_annotations = [anno for anno in self.raw_annotations if
                                       anno[1] >= min_test_id and anno[1] < test_end_id]
                print('test # random imgs', args.test_size)

            test_id_indices = set(np.random.choice(len(tmp_raw_annotations), size=args.test_size, replace=False))
            self.raw_annotations = [anno for idx, anno in enumerate(tmp_raw_annotations) if
                                    idx in test_id_indices]

    def __len__(self):
        return len(self.raw_annotations)

    def __getitem__(self, index):
        anno = self.raw_annotations[index]

        img_path = osp.join(self.root, anno[0])
        label = anno[1]
        landmarks = torch.empty((5, 2), dtype=torch.float32)
        for i in range(5):
            landmarks[i, 0] = anno[2 * i + 2]
            landmarks[i, 1] = anno[2 * i + 3]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label, landmarks, img_path

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of imgs: {}\n'.format(self.__len__())
        fmt_str += '    Is training: {}\n'.format(self.train)
        fmt_str += '    Is evaluating: {}\n'.format(self.evaluate)
        fmt_str += '    Is validation during training: {}\n'.format(self.val_during_training)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str
