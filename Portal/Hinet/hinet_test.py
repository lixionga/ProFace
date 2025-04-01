import random
import cv2
import numpy as np
import torch.nn

import torch.optim
from Read import load_embs_features

from img_utils import *
from torchvision.transforms import transforms
from Vector import vector_var
# from mobilefaceswap.image import Mobile_face
from model import *
# import basedatasets
from modules.Unet_common import *
from PIL import Image
import torchvision.transforms as T
from ESRGAN import *

from Mobile_Faceswap import Mobile_face
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

transform_val = T.Compose([
    # T.CenterCrop(c.cropsize_val),
    T.Resize([c.cropsize_val, c.cropsize_val]),
    T.ToTensor(),
])

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

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

def load(name):
    state_dicts = torch.load('/Users/mac/代码/test/Hinet/model/HiNet_patchGAN_model_checkpoint_00057.pt', map_location=torch.device('cpu'))
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    template_init.load_state_dict(state_dicts['template_init'])
    discriminator.load_state_dict(state_dicts['discriminator'])

def test():
    with torch.no_grad():
        net.eval()
        discriminator.eval()
        template_init.eval()
        data = Image.open('/Users/mac/代码/test/Hinet/test/30001.png')
        data = to_rgb(data)
        image = transform_val(data)
        image = image.unsqueeze(0)

        secret = template.repeat([image.size()[0], 1, 1, 1])

        cover_input = dwt(image)
        secret_input = dwt(secret)

        input_img = torch.cat((cover_input, secret_input), 1)
        # print(input_img)
        out = net(input_img)
        # print(out)
        output_steg = out.narrow(1, 0, 4 * 3)

        # 随机普通图片操作
        steg = iwt(output_steg)
        out_img = tesnor2cv(steg)
        save_path = '/Users/mac/代码/test/Hinet/out/out_img.jpg'
        # if cv2.imwrite(save_path, out_img):
        #     print('save success')
        # else:
        #     print('save failed')

        
        
        faceswap_img = faceswap.img_swap(out_img)
        faceswap_img = cv2.resize(faceswap_img, (256, 256))
        faceswap_img = cv2tensor(faceswap_img)
        # faceswap_img = faceswap_img.unsqueeze(0)


        steg_nm = steg/255
        steg_nm_test = dwt(steg_nm)
        # output_z = out.narrow(1, 4 * 3, out.shape[1] - 4 * 3)
        # print('-------------------------',output_z.shape)
        output_z = torch.randn(1, 12, 128, 128)
        output_z = gauss_noise(output_z.shape)

        steg_mp_input_val = dwt(faceswap_img)
        input_rev = torch.cat((steg_nm_test, output_z), 1)
        input_rev_mp_val = torch.cat((steg_mp_input_val, output_z), 1)
        output_image = net(input_rev, rev=True)
        output_image_mp = net(input_rev_mp_val, rev=True)

        secret_rev_val = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        secret_rev_val = iwt(secret_rev_val)

        secret_rev_mp_val = output_image_mp.narrow(1, 4 * c.channels_in,
                                                       output_image_mp.shape[1] - 4 * c.channels_in)
        secret_rev_mp_val = iwt(secret_rev_mp_val)

            #################
            # discriminator #
            #################

            # steg signal diff
        diff_nm_secret_val = secret - secret_rev_val
        diff_mp_secret_val = secret - secret_rev_mp_val

            # 真和假数据同时输入到预测模型中
        all_inputs = torch.cat((diff_nm_secret_val, diff_mp_secret_val), dim=0)
        # target_labels = torch.cat((valid_label_val, fake_label_val), dim=0)

        preds = discriminator(all_inputs).reshape(-1)

        print(preds)


        
        
        
        # print(steg)
if __name__ == '__main__':
    seed_torch(25)
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
    ])
    # net = HiNet()
    # load('/home/ysc/HiNet/model/final/FFHQ256-inv4-3fun-vector-1-2-1-0.3-8-seed25-resize0.5/HiNet_patchGAN_model_checkpoint_00057.pt')

    net = Model()
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids='cpu')

    discriminator = Discriminator(input_shape=(c.channels_in, c.cropsize, c.cropsize))
    
    template_init = vector_var(size=256)
    optim_template = torch.optim.Adam(template_init.parameters(), lr=0.0001, weight_decay=1e-5)

    # 使用ImageFolder加载数据集 初始化target datasets
    # target_dir = c.TARGET_PATH
    # target_dataset = datasets.ImageFolder(root=target_dir, transform=data_transform)

    template = template_init()

    # test
    valid_label_val = torch.ones(1, dtype=torch.long, requires_grad=False)# 真实标签，都是1
    fake_label_val = torch.zeros(1, dtype=torch.long, requires_grad=False)# 假标签，都是0
    
    dwt = DWT()
    iwt = IWT()
    id_emb_list,id_feature_list = load_embs_features()
    faceswap = Mobile_face(id_emb_list[0],id_feature_list[0])
    load('/Users/mac/代码/test/Hinet/model/HiNet_patchGAN_model_checkpoint_00057.pt')
    test()