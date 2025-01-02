import os
import time

import torchvision
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from options.demo_options import DemoOptions
from models.pix2pix_model import Pix2PixModel
from models.networks.sync_batchnorm import DataParallelWithCallback

opt = DemoOptions().parse()

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

def denorm(tensor):
    device = tensor.device
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

model = Pix2PixModel(opt)

if len(opt.gpu_ids) > 0:
            model = DataParallelWithCallback(model,device_ids=opt.gpu_ids)
model.eval()

if not os.path.exists(opt.result_dir):
    os.mkdir(opt.result_dir)

list1=[]
import data
dataloader = iter(data.create_dataloader(opt))
for i in range(opt.demo_nums):
    print(i,'/',opt.demo_nums,' demo')

    if(i==5):
        print(len(list1))
        demo_image_batch = torch.cat(list1,dim=2)
        save_image(demo_image_batch, f'{opt.result_dir}/demo_batch.jpg')


    input_data = []
    input_data.append(next(dataloader))
    if opt.demo_mode == 'partial':
        input_data.append(next(dataloader))
        input_data.append(next(dataloader))
    elif opt.demo_mode == 'multiple_refs':
        input_data.append(next(dataloader))
        input_data.append(next(dataloader))
        input_data.append(next(dataloader))
    elif opt.demo_mode not in ['normal', 'removal', 'interpolate','attack']:
        print('|demo_mode| is invalid!')
        break
    time_start = time.time()
    outs = model(input_data, mode='inference')
    print(outs[0].device)
    outs = [outs[i].cpu() for i in range(len(outs))]
    time_end = time.time()
    # print(time_end - time_start)

    opt.output_name = f'{opt.result_dir}/{i}'
    
    c = denorm(input_data[0]['nonmakeup'])
    s0 = denorm(input_data[0]['makeup'])

    if opt.demo_mode == 'normal':
        demo = torch.cat([c, s0, denorm(outs[0])], 3)
        save_image(denorm(outs[0]), f'{opt.output_name}_result.jpg')
        save_image(c, f'{opt.output_name}_source.jpg')
        save_image(s0, f'{opt.output_name}_reference.jpg')
        list1.append(demo)
        print(f'result saved into files starting with {opt.output_name}_demo.jpg\n')
        continue
    if opt.demo_mode == 'removal':
        demo = torch.cat([s0, c, denorm(outs[0]), denorm(outs[1]), denorm(outs[2])], 3)
        save_image(demo, f'{opt.output_name}_removal_demo.jpg')
        print(f'result saved into files starting with {opt.output_name}_removal_demo.jpg\n')
        continue
    if opt.demo_mode == 'interpolate':
        demo = torch.cat([c, s0, denorm(outs[0]), denorm(outs[1]), denorm(outs[2])], 3)
        save_image(demo, f'{opt.output_name}_interpolate_demo.jpg')
        print(f'result saved into files starting with {opt.output_name}_interpolate_demo.jpg\n')
        continue
    if opt.demo_mode == 'attack':
        demo = torch.cat([c, s0, denorm(input_data[0]['target']), denorm(outs[0])], 3)
        save_image(demo, f'{opt.output_name}_attack_demo.jpg')
        print(f'result saved into files starting with {opt.output_name}_attack_demo.jpg\n')
        continue
    s1 = denorm(input_data[1]['makeup'])
    s2 = denorm(input_data[2]['makeup'])
    if opt.demo_mode == 'partial':
        demo = torch.cat([c, s0, s1, s2,  denorm(outs[0])], 3)
        save_image(demo, f'{opt.output_name}_partial_demo.jpg')
        print(f'result saved into files starting with {opt.output_name}_partial_demo.jpg\n')
        continue
    s3 = denorm(input_data[3]['makeup'])
    if opt.demo_mode == 'multiple_refs':
        blank = torch.ones_like(s0)
        row1 = torch.cat([s0, denorm(outs[0]), denorm(outs[1]), denorm(outs[2]), s1], 3)
        row2 = torch.cat([blank, denorm(outs[3]), denorm(outs[4]), denorm(outs[5]), blank], 3)
        row3 = torch.cat([s2, denorm(outs[6]), denorm(outs[7]), denorm(outs[8]), s3], 3)
        demo = torch.cat([row1, row2, row3], 2)
        save_image(demo, f'{opt.output_name}_multiple_refs_demo.jpg')

    print(f'result saved into files starting with {opt.output_name}_multiple_refs_demo.jpg\n')
