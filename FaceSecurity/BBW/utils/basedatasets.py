import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import FaceSecurity.BBW.config.cfg as c
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
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        if mode == 'test':
            # print(c.TEST_PATH + "/*." + c.format_val)
            self.files = sorted(glob.glob(c.TEST_PATH + "/*." + c.format_val))
        if mode == 'val':
            # val
            self.files = sorted(glob.glob(c.VAL_PATH + "/*.png"))
        if mode == 'WID':
            # val
            self.files = sorted(glob.glob("/Data/CelebA-HQ/CelebA-HQ-256/Crop_256_watermarked/*.jpg"))
        if mode == 'VGG':
            # val
            self.files = sorted(glob.glob("/Data/vggface2-224/test_256/*.jpg"))

    def __getitem__(self, index):
        # try:
        #     landmark_file = self.files[index]
        #     landmark = np.load(landmark_file)
        #     image_path = landmark_file.replace("landmark", "image").replace(".npy", ".png")
        #     image = Image.open(image_path)
        #     image = to_rgb(image)
        #     item = self.transform(image)
        #     return item, landmark
        # except:
        #     return self.__getitem__(index + 1)

        if self.mode == 'train':
            try:
                landmark_file = self.files[index]
                landmark = np.load(landmark_file)
                image_path = landmark_file.replace("landmark", "image").replace(".npy", ".png")
                image = Image.open(image_path)
                image = to_rgb(image)
                item = self.transform(image)
                return item, landmark
            except:
                return self.__getitem__(index + 1)
        else:
            try:
                image_path = self.files[index]
                image = Image.open(image_path)
                image = to_rgb(image)
                item = self.transform(image)
                return item
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
    batch_size=c.batchsize_val,
    shuffle=False,
    num_workers=1,
    drop_last=True
)

# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode='test'),
    batch_size=c.batchsize_val,
    shuffle=False,
    num_workers=1,
    drop_last=True
)

# Test data loader
vggloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode='VGG'),
    batch_size=c.batchsize_val,
    shuffle=False,
    num_workers=1,
    drop_last=True
)

# Test data loader
WID_loader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode='WID'),
    batch_size=c.batchsize_val,
    shuffle=False,
    num_workers=1,
    drop_last=True
)
