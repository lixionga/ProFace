import os
import time
from PIL import Image
import sys

import torch.nn.functional as F
import torch
import torchvision.transforms as transforms

from torchvision.utils import save_image

from tqdm import tqdm
from Makeupprivacy.util.util import *
from Makeupprivacy.options.base_options import BaseOptions
from Makeupprivacy.models.pix2pix_model import Pix2PixModel
from Makeupprivacy.models.networks.sync_batchnorm import DataParallelWithCallback
from Makeupprivacy.models.networks.face_parsing.parsing_model import BiSeNet

class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--test_name', type=str, default='1_facenet_multiscale=2', help='Overridden description for test',dest='name')
        parser.add_argument("--source_dir", default="./Dataset-test/CelebA-HQ",help="path to source images")
        parser.add_argument("--reference_dir", default="./Dataset-test/reference",help="path to reference images")
        parser.add_argument('--which_epoch', type=str, default='latest',help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--beyond_mt', default='True',help='Want to transfer images that are not included in MT dataset, make sure this is Ture')
        parser.add_argument('--demo_mode', type=str, default='normal',help='normal|interpolate|removal|multiple_refs|partly')
        parser.add_argument("--save_path", default="/home/chenyidou/x_test/web/Makeup-privacy/imgs/save_img",help="path to source images")
        
        self.isTrain = False
        return parser
opt = TestOptions().parse()

# model = Pix2PixModel(opt)

# model.eval()

# n_classes = 19
# parsing_net = BiSeNet(n_classes=n_classes)
# parsing_net.load_state_dict(torch.load('/Users/mac/代码/Makeup-privacy/79999_iter.pth', map_location=torch.device('cpu')))
# parsing_net.eval()
# for param in parsing_net.parameters():
#     param.requires_grad = False

def denorm(tensor):
    device = tensor.device
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def tesnor2cv(img):
    # 移除批次维度
    img = img.squeeze(0).detach().cpu().numpy()
    # 转换为 (H, W, C) 格式
    img = np.transpose(img, (1, 2, 0))
    # 反归一化到 [0, 255]
    img *= 255
    img = img.astype(np.uint8)
    # 从 RGB 转为 BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def cv2tensor(img):
    # 从 BGR 转为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为 (C, H, W) 格式
    img = np.transpose(img, (2, 0, 1))
    # 转为 PyTorch 张量，并添加一个批次维度
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    # 归一化到 [0, 1]
    img = img / 255.0
    return img
class MakeupPrivacy:
    def __init__(self):
        model = Pix2PixModel(opt)

        model.eval()

        n_classes = 19
        parsing_net = BiSeNet(n_classes=n_classes)
        parsing_net.load_state_dict(torch.load('/home/chenyidou/x_test/web/Makeupprivacy/79999_iter.pth', map_location=torch.device('cpu')))
        parsing_net.eval()
        for param in parsing_net.parameters():
            param.requires_grad = False

        self.model = model
        self.parsing_net = parsing_net

    def forward(self, image,s_img):
        source_image = cv2.imread(image)
        img = cv2.imread(s_img)
        cm = cv2.resize(source_image, (512, 512))
        sm = cv2.resize(img, (512, 512))

        c = cv2.resize(source_image, (256, 256))
        s = cv2.resize(img, (256, 256))

        cm_tensor = cv2tensor(cm)
        sm_tensor = cv2tensor(sm)
        c_tensor = cv2tensor(c)
        s_tensor = cv2tensor(s)

        x_label = self.parsing_net(cm_tensor)[0]
        y_label = self.parsing_net(sm_tensor)[0]
        x_label = F.interpolate(x_label, (256, 256), mode='bilinear', align_corners=True)
        y_label = F.interpolate(y_label, (256, 256), mode='bilinear', align_corners=True)
        x_label = torch.softmax(x_label, 1)
        y_label = torch.softmax(y_label, 1)

        nonmakeup_unchanged = (x_label[0, 0, :, :] + x_label[0, 4, :, :] + x_label[0, 5, :, :] + x_label[0, 11, :,:] + x_label[0, 16,:,:] + x_label[0, 17, :,:]).unsqueeze(0).unsqueeze(0)
        makeup_unchanged = (y_label[0, 0, :, :] + y_label[0, 4, :, :] + y_label[0, 5, :, :] + y_label[0, 11, :,:] + y_label[0, 16, :,:] + y_label[0,17, :,:]).unsqueeze(0).unsqueeze(0)
        
        input_dict = {'nonmakeup': c_tensor,
                          'makeup': s_tensor,
                          'label_A': x_label,
                          'label_B': y_label,
                          'makeup_unchanged': makeup_unchanged,
                          'nonmakeup_unchanged': nonmakeup_unchanged
                          }
        
        synthetic_image = self.model([input_dict], mode='inference')

        out = tesnor2cv(synthetic_image[0])
        return out
    

if __name__ == '__main__':
    model = MakeupPrivacy()
    out = model.forward('/home/chenyidou/x_test/web/Makeupprivacy/imgs/source_img/000291.jpg')
    cv2.imwrite('/home/chenyidou/x_test/web/Makeupprivacy/imgs/save_img/000001.jpg', out)
