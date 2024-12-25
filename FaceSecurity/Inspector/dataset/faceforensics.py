import torch
import numpy as np
from os.path import join
from dataset import AbstractDataset

METHOD = ['all', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
SPLIT = ['train', 'train_real', 'train_fake', 'val', 'val_real', 'val_fake', 
         'test', 'test_real', 'test_fake']
COMP2NAME = {'c0': 'raw', 'c23': 'c23', 'c40': 'c40'}
SOURCE_MAP = {'youtube': 2, 'Deepfakes': 3, 'Face2Face': 4, 'FaceSwap': 5, 'NeuralTextures': 6}


class FaceForensics(AbstractDataset):
    """
    FaceForensics++ Dataset proposed in "FaceForensics++: Learning to Detect Manipulated Facial Images"
    """

    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if cfg['split'] not in SPLIT:
            raise ValueError(f"split should be one of {SPLIT}, "
                             f"but found {cfg['split']}.")
        if cfg['method'] not in METHOD:
            raise ValueError(f"method should be one of {METHOD}, "
                             f"but found {cfg['method']}.")
        if cfg['compression'] not in COMP2NAME.keys():
            raise ValueError(f"compression should be one of {COMP2NAME.keys()}, "
                             f"but found {cfg['compression']}.")
        super(FaceForensics, self).__init__(
            cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'FF++ {cfg['method']}' of split '{cfg['split']}' "
              f"and compression '{cfg['compression']}'\nPlease wait patiently...")
        
        self.categories = ['original', 'fake']
        # load the path of dataset images
        indices = join(self.root, cfg['split'] + "_" + cfg['compression'] + ".pickle")
        indices = torch.load(indices)
        if cfg['method'] == "all":
            # full dataset
            self.images = [join(cfg['dataroot'], _[0]) for _ in indices] # img_path
            self.targets = [_[1] for _ in indices] # tgt_lab
        else:
            # specific manipulated method
            self.images = list()
            self.targets = list()
            nums = 0
            for _ in indices:
                if cfg['method'] in _[0]:
                    self.images.append(join(cfg['root'], _[0]))
                    self.targets.append(_[1])
                nums = len(self.targets)
            ori = list()
            for _ in indices:
                if "original_sequences" in _[0]:
                    ori.append(join(cfg['root'], _[0]))
            choices = np.random.choice(ori, size=nums, replace=False)
            self.images.extend(choices)
            self.targets.extend([0] * nums)
        print("Data from 'FF++' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.\n")
