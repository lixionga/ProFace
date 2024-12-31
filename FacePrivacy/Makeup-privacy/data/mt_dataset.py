import os.path
from torchvision.utils import save_image
import torchvision.transforms as transforms

from PIL import Image
import PIL
import numpy as np
import torch
from torch.autograd import Variable
import os, sys
os.chdir(sys.path[0])

def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


class MTDataset():
    def __init__(self, opt):
        self.random = None
        self.phase=opt.phase
        self.opt = opt
        self.root = opt.dataroot
        self.dir_makeup = opt.dataroot  
        self.dir_nonmakeup = opt.dataroot
        self.dir_seg = opt.dirmap  
        self.n_componets = opt.n_componets
        self.makeup_names = []
        self.non_makeup_names = []
        
        if self.phase == 'train':
            self.makeup_names = [name.strip() for name in
                                 open(os.path.join('MT-Dataset', 'makeup.txt'), "rt").readlines()]
            self.non_makeup_names = [name.strip() for name in
                                     open(os.path.join('MT-Dataset', 'non-makeup.txt'), "rt").readlines()]
        if self.phase == 'test':
            self.makeup_names = [name.strip() for name in
                                 open(os.path.join('MT-Dataset-demo', 'makeup_test.txt'), "rt").readlines()]
            self.non_makeup_names = [name.strip() for name in
                                     open(os.path.join('MT-Dataset-demo', 'non-makeup_test.txt'), "rt").readlines()]

        self.transform = transforms.Compose([
            transforms.Resize((opt.crop_size, opt.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize((opt.crop_size, opt.crop_size), interpolation=PIL.Image.NEAREST),
            ToTensor])

    def __getitem__(self, index):
        if self.phase == 'test':
            makeup_name = self.makeup_names[index]
            nonmakeup_name = self.non_makeup_names[index]
            # target_img
            if self.opt == 'attack':
                target_img=Image.open(self.target_path).convert('RGB')
                target_img = self.transform(target_img)
                

        if self.phase == 'train':
            index = self.pick()
            makeup_name = self.makeup_names[index[0]]
            nonmakeup_name = self.non_makeup_names[index[1]]

        nonmakeup_path = os.path.join(self.dir_nonmakeup, nonmakeup_name)
        makeup_path = os.path.join(self.dir_makeup, makeup_name)

        # print(makeup_path)

        makeup_img = Image.open(makeup_path).convert('RGB')
        nonmakeup_img = Image.open(nonmakeup_path).convert('RGB')

        

        makeup_seg_img = Image.open(os.path.join(self.dir_seg, makeup_name))
        nonmakeup_seg_img = Image.open(os.path.join(self.dir_seg,nonmakeup_name))

        makeup_img = self.transform(makeup_img)
        nonmakeup_img = self.transform(nonmakeup_img)
        mask_B = self.transform_mask(makeup_seg_img)  
        mask_A = self.transform_mask(nonmakeup_seg_img)

        label_B=mask_B
        label_A=mask_A

        makeup_unchanged = (mask_B == 0).float()  + (mask_B == 2).float() + (mask_B == 3).float() + (mask_B == 4).float() + (mask_B == 5).float() +\
                            (mask_B == 8).float() + (mask_B == 10).float() + (mask_B == 14).float()
        nonmakeup_unchanged = (mask_A == 0).float()  + (mask_A == 2).float() + (mask_A == 3).float()+ (mask_A == 4).float() + (mask_A == 5).float() +\
                                (mask_A == 8).float() + (mask_A == 10).float() + (mask_B == 14).float()

        mask_A_lip = (mask_A == 9).float() + (mask_A == 7).float()
        mask_B_lip = (mask_B == 9).float() + (mask_B == 7).float()
        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)

        mask_A_skin = (mask_A == 1).float() + (mask_A == 6).float() + (mask_A == 11).float() + (mask_A == 12).float()+ (mask_A == 13).float()
        mask_B_skin = (mask_B == 1).float() + (mask_B == 6).float()+ (mask_B == 11).float() + (mask_B == 12).float()+ (mask_B == 13).float()
        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)

        mask_A_eye_left = (mask_A == 5).float()
        mask_A_eye_right = (mask_A == 4).float()
        mask_B_eye_left = (mask_B == 5).float()
        mask_B_eye_right = (mask_B == 4).float()
        mask_A_face = (mask_A == 1).float() + (mask_A == 6).float()
        mask_B_face = (mask_B == 1).float() + (mask_B == 6).float()
        
      
        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
            self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
            self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)

        mask_A = {}
        mask_A["mask_A_eye_left"] = mask_A_eye_left
        mask_A["mask_A_eye_right"] = mask_A_eye_right
        mask_A["index_A_eye_left"] = index_A_eye_left
        mask_A["index_A_eye_right"] = index_A_eye_right
        mask_A["mask_A_skin"] = mask_A_skin
        mask_A["index_A_skin"] = index_A_skin
        mask_A["mask_A_lip"] = mask_A_lip
        mask_A["index_A_lip"] = index_A_lip

        mask_B = {}
        mask_B["mask_B_eye_left"] = mask_B_eye_left
        mask_B["mask_B_eye_right"] = mask_B_eye_right
        mask_B["index_B_eye_left"] = index_B_eye_left
        mask_B["index_B_eye_right"] = index_B_eye_right
        mask_B["mask_B_skin"] = mask_B_skin
        mask_B["index_B_skin"] = index_B_skin
        mask_B["mask_B_lip"] = mask_B_lip
        mask_B["index_B_lip"] = index_B_lip

        
                



        return {'mask_A': mask_A, 'mask_B': mask_B,
                'label_A':label_A, 'label_B':label_B,
                'nonmakeup': nonmakeup_img, 'makeup': makeup_img,
                'makeup_unchanged': makeup_unchanged, 'nonmakeup_unchanged': nonmakeup_unchanged,
                }

    def pick(self):
        if self.random is None:
            self.random = np.random.RandomState(np.random.seed())
        a_index = self.random.randint(0, len(self.makeup_names))
        another_index = self.random.randint(0, len(self.non_makeup_names))
        return [a_index, another_index]

    def __len__(self):
        return len(self.non_makeup_names)

    def name(self):
        return 'MT-Dataset'


    def rebound_box(self, mask_A, mask_B, mask_A_face):
        mask_A = mask_A.unsqueeze(0)
        mask_B = mask_B.unsqueeze(0)
        mask_A_face = mask_A_face.unsqueeze(0)

        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[:, :, min(x_A_index) - 6:max(x_A_index) + 7, min(y_A_index) - 8:max(y_A_index) + 9] = \
            mask_A_face[:, :, min(x_A_index) - 6:max(x_A_index) + 7, min(y_A_index) - 8:max(y_A_index) + 9]
        mask_B_temp[:, :, min(x_B_index) - 6:max(x_B_index) + 7, min(y_B_index) - 8:max(y_B_index) + 9] = \
            mask_A_face[:, :, min(x_B_index) - 6:max(x_B_index) + 7, min(y_B_index) - 8:max(y_B_index) + 9]
        mask_A_temp = mask_A_temp.squeeze(0)
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        mask_A_face = mask_A_face.squeeze(0)
        mask_B_temp = mask_B_temp.squeeze(0)

        return mask_A_temp, mask_B_temp

    def mask_preprocess(self, mask_A, mask_B): 
        mask_A = mask_A.unsqueeze(0)  
        mask_B = mask_B.unsqueeze(0)
        index_tmp = torch.nonzero(mask_A, as_tuple=False) 
        x_A_index = index_tmp[:, 2]  

        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        return mask_A, mask_B, index, index_2

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(batchSize=1)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='normal')
        return parser

class MTDataLoader():
    def __init__(self, opt):
        self.dataset = MTDataset(opt)
        print("Dataset loaded")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
            )


    def name(self):
        return 'MTDataLoader'


    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
