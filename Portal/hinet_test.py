import random
import sys
import warnings
import cv2
import kornia
import numpy as np
import torch.nn
import torch.nn.functional as F
import torch.optim
import os
# from nvidia import cudnn
from sklearn.metrics import roc_auc_score

from Hinet.img_utils import *
from lpips import lpips
from torchvision.transforms import transforms
from Hinet.Vector import vector_var
# from mobilefaceswap.image import Mobile_face
from Hinet.model import *
# import Hinet.basedatasets
from Hinet.modules.Unet_common import *
from PIL import Image
import torchvision.transforms as T
from Hinet.ESRGAN import *
from Read import load_embs_features
from Mobile_Faceswap import Mobile_face
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

def seed_torch(seed=25):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    # cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)

def gauss_noise(shape):
    noise = torch.zeros(shape)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape)
    return noise

class Hinet_model:
    def __init__(self):
        seed_torch(25)
        net = Model()
        init_model(net)
        net.to("cpu")
        net = torch.nn.DataParallel(net)
        discriminator = Discriminator(input_shape=(c.channels_in, c.cropsize, c.cropsize))
    
        
        template_init = vector_var(size=256)
        state_dicts = torch.load('/home/chenyidou/x_test/web/Hinet/model/HiNet_patchGAN_model_checkpoint_00057.pt', map_location=torch.device('cpu'), weights_only=True)
        network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
        net.load_state_dict(network_state_dict)
        template_init.load_state_dict(state_dicts['template_init'])
        discriminator.load_state_dict(state_dicts['discriminator'])

        self.dwt = DWT()
        self.iwt = IWT()
        self.template = template_init()
        self.net = net.eval()
        self.discriminator = discriminator.eval()

    def img_make(self, image):
        img = cv2.imread(image)
        img = cv2.resize(img, (256,256))
        img = cv2tensor(img)
        secret = self.template.repeat([img.size()[0], 1, 1, 1])
        # cv2.imwrite('/home/chenyidou/x_test/web/Hinet/out/forward.jpg',tesnor2cv(secret))
        cover_input = self.dwt(img)
        secret_input = self.dwt(secret)

        input_img = torch.cat((cover_input, secret_input), 1)
        # print(input_img)
        out = self.net(input_img)
        # print(out)
        output_steg = out.narrow(1, 0, 4 * 3)

        # 随机普通图片操作
        steg = self.iwt(output_steg)
        out_img = tesnor2cv(steg)
        return out_img,tesnor2cv(secret)
    
    def img_check(self,image):
        img = cv2.imread(image)
        img = cv2.resize(img, (256,256))
        img = cv2tensor(img)
        
        
        steg = self.dwt(img)
        secret = self.template.repeat([img.size()[0], 1, 1, 1])
        output_z = torch.randn(1, 12, 128, 128)
        output_z = gauss_noise(output_z.shape)

        # steg_mp_input_val = self.dwt(img1)
        input_rev = torch.cat((steg, output_z), 1)
        output_image = self.net(input_rev, rev=True)

        secret_rev_val = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        secret_rev_val = self.iwt(secret_rev_val)



            # steg signal diff
        diff_nm_secret_val = secret
        diff_mp_secret_val = secret - secret_rev_val

            # 真和假数据同时输入到预测模型中
        all_inputs = torch.cat((diff_nm_secret_val, diff_mp_secret_val), dim=0)
        # target_labels = torch.cat((valid_label_val, fake_label_val), dim=0)

        preds = self.discriminator(all_inputs).reshape(-1)
        return preds,tesnor2cv(secret),tesnor2cv(secret_rev_val)
    
    
    def forward(self,image):
        img = cv2.imread(image)
        img = cv2.resize(img, (256,256))
        img = cv2tensor(img)
        secret = self.template.repeat([img.size()[0], 1, 1, 1])

        cover_input = self.dwt(img)
        secret_input = self.dwt(secret)

        input_img = torch.cat((cover_input, secret_input), 1)
        out = self.net(input_img)
        output_steg = out.narrow(1, 0, 4 * 3)

        steg = self.iwt(output_steg)
        out_img = tesnor2cv(steg)


        faceswap_img = faceswap.hinet_swap(out_img)

        faceswap_img = cv2.resize(out_img,(256,256)) #  伪造输入
        faceswap_img = cv2tensor(faceswap_img)

        steg_nm = steg#真实输入
        steg_nm_test = self.dwt(steg_nm)
        # output_z = out.narrow(1, 4 * 3, out.shape[1] - 4 * 3)
        # print('-------------------------',output_z.shape)
        output_z = torch.randn(1, 12, 128, 128)
        output_z = gauss_noise(output_z.shape)

        steg_mp_input_val = self.dwt(faceswap_img)
        input_rev = torch.cat((steg_nm_test, output_z), 1)
        input_rev_mp_val = torch.cat((steg_mp_input_val, output_z), 1)
        output_image = self.net(input_rev, rev=True)
        output_image_mp = self.net(input_rev_mp_val, rev=True)

        secret_rev_val = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        secret_rev_val = self.iwt(secret_rev_val)
        # cv2.imwrite('/home/chenyidou/x_test/web/Hinet/output/real_backword1.jpg',tesnor2cv(secret_rev_val))
        secret_rev_mp_val = output_image_mp.narrow(1, 4 * c.channels_in,
                                                       output_image_mp.shape[1] - 4 * c.channels_in)
        secret_rev_mp_val = self.iwt(secret_rev_mp_val)
        # cv2.imwrite('/home/chenyidou/x_test/web/Hinet/output/fake_backword1.jpg',tesnor2cv(secret_rev_mp_val))

            #################
            # discriminator #
            #################

            # steg signal diff
        diff_nm_secret_val = secret - secret_rev_val
        diff_mp_secret_val = secret - secret_rev_mp_val

            # 真和假数据同时输入到预测模型中
        all_inputs = torch.cat((diff_nm_secret_val, diff_mp_secret_val), dim=0)
        # target_labels = torch.cat((valid_label_val, fake_label_val), dim=0)

        preds = self.discriminator(all_inputs).reshape(-1)

        # print(preds)
        return preds
    def check(self,img1,img2):
        img = cv2.imread(img1)
        img = cv2.resize(img,(256,256))
        img = cv2tensor(img)

        img3 = cv2.imread(img2)
        img3 = cv2.resize(img3,(256,256))
        img3 = cv2tensor(img3)

        secret = self.template.repeat([img.size()[0], 1, 1, 1])

        cover_input = self.dwt(img3)
        secret_input = self.dwt(secret)

        input_img = torch.cat((cover_input, secret_input), 1)
        out = self.net(input_img)
        steg = out.narrow(1, 0, 4 * 3)

        output_z = torch.randn(1, 12, 128, 128)
        output_z = gauss_noise(output_z.shape)

        steg_mp_input_val = self.dwt(img)
        input_rev = torch.cat((steg.cpu(), output_z), 1)

        input_rev_mp_val = torch.cat((steg_mp_input_val, output_z), 1)
        output_image = self.net(input_rev, rev=True)
        output_image_mp = self.net(input_rev_mp_val, rev=True)

        secret_rev_val = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        secret_rev_val = self.iwt(secret_rev_val)
        # cv2.imwrite('/home/chenyidou/x_test/web/Hinet/output/real_backword1.jpg',tesnor2cv(secret_rev_val))
        secret_rev_mp_val = output_image_mp.narrow(1, 4 * c.channels_in,
                                                       output_image_mp.shape[1] - 4 * c.channels_in)
        secret_rev_mp_val = self.iwt(secret_rev_mp_val)

        diff_nm_secret_val = secret - secret_rev_val
        diff_mp_secret_val = secret - secret_rev_mp_val

            # 真和假数据同时输入到预测模型中
        all_inputs = torch.cat((diff_nm_secret_val, diff_mp_secret_val), dim=0)
        # target_labels = torch.cat((valid_label_val, fake_label_val), dim=0)

        preds = self.discriminator(all_inputs).reshape(-1)
        return preds

if __name__ == '__main__':
    id_emb_list,id_feature_list = load_embs_features()
    faceswap = Mobile_face(id_emb_list[0],id_feature_list[0])
    hinet_model = Hinet_model()
    # out = hinet_model.img_make('/home/chenyidou/x_test/web/Hinet/test/30000.png')
    # out = hinet_model.check('/home/chenyidou/x_test/Hinet/out/out_img.jpg','/home/chenyidou/x_test/Hinet/out/processed_1739862100_out_img.jpg')
    out = hinet_model.forward('/home/chenyidou/x_test/web/demo_file/gtx.jpg')
    print(out)
    # cv2.imwrite('/home/chenyidou/x_test/web/Hinet/out/out.jpg',out)
        