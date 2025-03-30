from templates import *
import matplotlib.pyplot as plt
import os
import torch as th
import random

# 扩散模型
device = 'cuda:0'       # TODO: 默认使用GPU
conf = ffhq128_autoenc_130M()
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

# 配置文件
T_inv = 100
T_step = 100
e_conf = {
    'is_recovery': False,
    'is_use': True,       # 是否使用该配置（是否保持原模型流程不变）
    'inner_t': 650,       # TODO: x0加噪到第几步 - [0, 1000]
    'start': 600,         # TODO: 何时开始引导 - [0, 1000]
    'stop': 100,          # 何时结束引导 - [0, 1000]
    'rho_scale': 0.05,    # TODO: 匿名强度
    'fun_type': 'idis',   # 引导函数类型：idis
    'gen_batch_size': 4,  # TODO: 1张图像生成多少张图像
    'optim_targ': ['xt', 'cond'][1]
}
e_conf['use_inner_mode'] = e_conf['xT_mode'] == 1  # xT编码模式是否使用1

assert e_conf['fun_type'] in ['idis'], 'fun_type error, 匿名化引导时，值必须是 idis'
assert 4 >= e_conf['gen_batch_size'] > 0, '生成数量太多，必须小于4'

# 数据
data_path = 'datas'      # TODO: 测试图像
if not os.path.exists(data_path):
    os.makedirs(data_path)
dataset = ImageDataset(data_path, image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
e_conf['ref_path'] = dataset[0]['path']  # 参考图像（先随便放一个，后面会动态更换）

assert dataloader.batch_size == 1, "保证每个批次只有一张原始图像"

# 身份向量提取模型
from PIL import Image
from functions.arcface_torch.model import IDLoss
idloss = IDLoss(ref_path=e_conf["ref_path"]).cuda()
idloss.eval()
idloss.to(device)
e_conf['idloss'] = idloss

# 开始
output_path = 'outputs/test_anony/'  # TODO: 输出路径
if not os.path.exists(output_path):
    os.makedirs(output_path)
with tqdm(total=len(dataloader), desc=f"匿名进度") as progress_bar:
    for i, batch in enumerate(dataloader):
        imgs, idxs, img_paths = batch['img'], batch['index'], batch['path']
        idloss.change_ref_image(img_paths[0])
        img_name = ((img_paths[0].split("/"))[-1]).split(".")[0]
        # 编码
        cond = model.encode(imgs.to(device))
        cond = cond.repeat(e_conf['gen_batch_size'], 1)
        # 反演/加噪到中间时步
        imgs = imgs.repeat(e_conf['gen_batch_size'], 1, 1, 1)
        xT = model.sample_xt_from_x0(imgs.to(device), T=1000, t=e_conf['inner_t'])
        # 解码/梯度引导匿名化
        pred = model.render(xT, cond, T=T_step, energy_fun_conf=e_conf)  # [0, 1]
        # 保存
        res_len = pred.shape[0]
        ims = th.clamp(pred, min=0.0, max=1.0)
        for j in range(res_len):
            im = Image.fromarray((ims[j].permute(1, 2, 0).cpu() * 255).numpy().astype('uint8'))
            im.save(os.path.join(output_path, f'{img_name}_{j}.png'))

        progress_bar.update(1)

