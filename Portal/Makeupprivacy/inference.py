import os
import time
from PIL import Image
import sys

import torch.nn.functional as F
import torch
import torchvision.transforms as transforms

from torchvision.utils import save_image

from tqdm import tqdm
from util.util import *
from options.base_options import BaseOptions
from models.pix2pix_model import Pix2PixModel
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.networks.face_parsing.parsing_model import BiSeNet


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--test_name', type=str, default='1_facenet_multiscale=2', help='Overridden description for test',dest='name')
        parser.add_argument("--source_dir", default="./Dataset-test/CelebA-HQ",help="path to source images")
        parser.add_argument("--reference_dir", default="./Dataset-test/reference",help="path to reference images")
        parser.add_argument('--which_epoch', type=str, default='latest',help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--beyond_mt', default='True',help='Want to transfer images that are not included in MT dataset, make sure this is Ture')
        parser.add_argument('--demo_mode', type=str, default='normal',help='normal|interpolate|removal|multiple_refs|partly')
        parser.add_argument("--save_path", default="/Users/mac/代码/Makeup-privacy/imgs/save_img",help="path to source images")
        
        self.isTrain = False
        return parser
opt = TestOptions().parse()

# device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids[0] >= 0 else torch.device('cpu')

model = Pix2PixModel(opt)

if len(opt.gpu_ids) > 0:
            model = DataParallelWithCallback(model,device_ids=opt.gpu_ids)
model.eval()

n_classes = 19
parsing_net = BiSeNet(n_classes=n_classes)
parsing_net.load_state_dict(torch.load('/Users/mac/代码/Makeup-privacy/79999_iter.pth', map_location=torch.device('cpu')))
parsing_net.eval()
for param in parsing_net.parameters():
    param.requires_grad = False



def denorm(tensor):
    device = tensor.device
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])    
trans = transforms.Compose([transforms.ToTensor(),normalize])

def generate():
    source_paths = os.listdir(source_dir)
    source_paths.sort(key=lambda x: int(x.split('.')[0]))  
    reference_paths = os.listdir(reference_dir)
    for source_path in tqdm(source_paths):
        source_name = source_path.replace('.jpg', '.png')
        source_path = os.path.join(source_dir, source_path)
        c= Image.open(source_path).convert("RGB")
        for reference_path in reference_paths:
            reference_name = reference_path.split('.')[0]
            reference_path = os.path.join(reference_dir, reference_path)
            s = Image.open(reference_path).convert("RGB")
            height, width = c.size[0], c.size[1]
            c_m = c.resize((512, 512))
            s_m = s.resize((512, 512))
            c = c.resize((256, 256))
            s = s.resize((256, 256))
            # print(c.size)
            c_tensor = trans(c).unsqueeze(0)
            s_tensor = trans(s).unsqueeze(0)
            c_m_tensor = trans(c_m).unsqueeze(0)
            s_m_tensor = trans(s_m).unsqueeze(0)

            x_label = parsing_net(c_m_tensor)[0]
            y_label = parsing_net(s_m_tensor)[0]
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
            synthetic_image = model([input_dict], mode='inference')
            out = denorm(synthetic_image[0])
            out = F.interpolate(out, (256, 256 * height // width), mode='bilinear', align_corners=False)
            save_path = os.path.join(opt.save_path, reference_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = os.path.join(opt.save_path, reference_name, source_name)
            save_image(out, f'{save_name}')
    print('Finished! Image saved in:', os.path.abspath("/home/chenyidou/x_test/project/model/Makeup-privacy/imgs/save_img"))


if __name__ == '__main__':
    source_dir="/Users/mac/代码/Makeup-privacy/imgs/source_img"
    reference_dir="/Users/mac/代码/Makeup-privacy/imgs/reference"
    generate()