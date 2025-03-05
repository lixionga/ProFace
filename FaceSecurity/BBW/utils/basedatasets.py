import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import FaceSecurity.BBW.config.config as c
from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_landmark)))
        if mode == 'test':
            self.files = sorted(glob.glob(c.TEST_PATH + "/*." + c.format_img))
        if mode == 'val':
            # val
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_img))
        if mode == 'ganimation':
            self.files = sorted(glob.glob(c.GANimation_DATA_PAHT + "/*." + c.format_img))
        if mode == 'WID':
            self.files = sorted(glob.glob(c.WID_DATA_PAHT + "/*.jpg"))
        if mode == 'VGG':
            self.files = sorted(glob.glob("/Data/vggface2-224/test_256/*.jpg"))

    def __getitem__(self, index):

        if self.mode != 'train':
            try:
                image_path = self.files[index]
                image = Image.open(image_path)
                image = to_rgb(image)
                item = self.transform(image)
                return item
            except:
                return self.__getitem__(index + 1)

        else:
            try:
                landmark_file = self.files[index]
                landmark = np.load(landmark_file)

                image_path = landmark_file.replace("landmark", "image").replace(c.format_landmark, c.format_img)

                image = Image.open(image_path)
                image = to_rgb(image)
                item = self.transform(image)

                return item, landmark

            except:
                return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)


transform = T.Compose([
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    # T.CenterCrop(c.cropsize_val),
    T.Resize([c.cropsize_val, c.cropsize_val]),
    T.ToTensor(),
])

# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode='train'),
    batch_size=c.batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True
)
# val data loader
valloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode='val'),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=True
)

# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode='test'),
    batch_size=1,
    shuffle=True,
    num_workers=1,
    drop_last=True
)

# GANimation data loader
# ganimationloader = DataLoader(
#     Hinet_Dataset(transforms_=transform_val, mode='ganimation'),
#     batch_size=c.batchsize_test,
#     shuffle=False,
#     num_workers=1,
#     drop_last=True
# )

# WIDloader data loader
WIDloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode='WID'),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=True
)

# WIDloader data loader
VGGloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode='VGG'),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=True
)
