# -*- encoding: utf-8 -*-
import sys
sys.path.append('..')
from model import BiSeNet
import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import  torch

import torchvision
from torchvision.transforms import Resize
totensor = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
resize_128 = Resize([128,128])


# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map

def vis_parsing_maps(im, parsing_anno,x,y, stride):
    # Colors for all 20 parts
    # part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
    #                [255, 0, 85], [255, 0, 170],
    #                [0, 255, 0], [85, 255, 0], [170, 255, 0],
    #                [0, 255, 85], [0, 255, 170],
    #                [0, 0, 255], [85, 0, 255], [170, 0, 255],
    #                [0, 85, 255], [0, 170, 255],
    #                [255, 255, 0], [255, 255, 85], [255, 255, 170],
    #                [255, 0, 255], [255, 85, 255], [255, 170, 255],
    #                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    part_colors = [[10, 10, 10], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [10, 10, 10],
                   [10, 10, 10], [10, 10, 10],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [10, 10, 10],
                   [10, 10, 10], [10, 10, 10], [10, 10, 10],
                   [10, 10, 10], [10, 10, 10], [10, 10, 10],
                   [10, 10, 10], [10, 10, 10], [10, 10, 10]]

    im = np.array(im)
    
    vis_parsing_anno = parsing_anno.copy()
    vis_parsing_anno_color = np.zeros((im.shape[0], im.shape[1], 3)) + 0

    face_mask = np.zeros((im.shape[0], im.shape[1]))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)# 获得对应分类的的像素坐标

        idx_y = (index[0]+y).astype(np.int32)
        idx_x = (index[1]+x).astype(np.int32)

        # continue
        vis_parsing_anno_color[idx_y,idx_x, :] = part_colors[pi]# 给对应的类别的掩码赋值

        face_mask[idx_y,idx_x] = 1
        # if pi in[1,2,3,4,5,6,7,8,10,11,12,13,14,17]:
        #     face_mask[idx_y,idx_x] = 0.35

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    face_mask = np.expand_dims(face_mask, 2)
    vis_im = vis_parsing_anno_color*face_mask
    vis_im = vis_im.astype(np.uint8)

    return vis_im


def inference( img_size, image_path, model_path):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()

    print('model : {}'.format(model_path))
    net.load_state_dict(torch.load(model_path))
    net.eval()
    test_idx = 0
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        idx = 0
        for f_ in os.listdir(image_path):
            img_ = cv2.imread(image_path + f_)
            img = Image.fromarray(cv2.cvtColor(img_,cv2.COLOR_BGR2RGB))

            image = img.resize((img_size, img_size))
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # print(f'img shape is {img.shape}')
            img = img.cuda()
            out = net(img)[0]
            parsing_ = out.squeeze(0).cpu().numpy().argmax(0)
            
            idx += 1
            print(f'parsing_ shape is {parsing_.shape}, and parsing_ type is {type(parsing_)}')
            print('<{}> image : '.format(idx),np.unique(parsing_))

            parsing_ = cv2.resize(parsing_,(img_.shape[1],img_.shape[0]),interpolation=cv2.INTER_NEAREST)

            parsing_ = parsing_.astype(np.uint8)
            vis_im = vis_parsing_maps(img_, parsing_, 0,0,stride=1)
            print(f'vis_im shape is {vis_im.shape}')
            # 保存输出结果
            test_result = './result/'
            if not osp.exists(test_result):
                os.makedirs(test_result)
            cv2.imwrite(test_result+"p_{}-".format(img_size)+f_,vis_im)

            # cv2.namedWindow("vis_im",0)
            # cv2.imshow("vis_im",vis_im)
            
            if cv2.waitKey(500) == 27:
                break
            
            

            # raw_img = cv2.imread('/home/yl/lk/code/ID-dise4/out/INN/TestDataOrin'+'/test_img'+str(test_idx)+'.png')
            # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            # raw_img = cv2.resize(raw_img,(256,256),interpolation=cv2.INTER_CUBIC)
            # raw_img = totensor(raw_img)
            # raw_img = raw_img.unsqueeze(0).to('cuda:0')

            # # manual_image_path = os.path.join(opts.image_dir,'/TestDataDistan', 'test_img',test_idx,'.png')
            # manual_img = cv2.imread('/home/yl/lk/code/ID-dise4/out/INN/TestDataDistan'+'/test_img'+str(test_idx)+'.png')
            # manual_img = cv2.cvtColor(manual_img, cv2.COLOR_BGR2RGB)
            # manual_img = cv2.resize(manual_img,(256,256),interpolation=cv2.INTER_CUBIC)
            # manual_img = totensor(manual_img)
            # manual_img = manual_img.unsqueeze(0).to('cuda:0')


            
            test_idx =int(test_idx)
            test_idx = test_idx +1


def get_fixed_imgs(bg_imgs, fg_imgs, model_path, device):

    """
    bg_imgs: 接收
    """
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net = net.to(device)

    # print('model : {}'.format(model_path))
    net.load_state_dict(torch.load(model_path))
    net.eval()
    trans = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        #取背景为mask
        img = trans(fg_imgs)
        # print(f'img shape is {img.shape}')
        out = net(img)[0]
        # print(f'out shape is {out.shape}, out type is {type(out)}')
        # print(f'out shape is {out.squeeze(0).argmax(0).shape}, out type is {type(out)}')
        # idx += 1
        # parsing_ = parsing_.type_as(torch.uint8)
        # print(f'parsing shape is {parsing_.shape}')
        # print(f'parsing_ is {parsing_}')

        
        #按颜色分割
        part_colors = torch.tensor([[0, 0, 0], [255, 255, 255], [255, 255, 255],
                    [255, 255, 255], [255, 255, 255],
                    [255, 255, 255], [255, 255, 255], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0],
                    [255, 255, 255], [255, 255, 255], [255, 255, 255],
                    [255, 255, 255], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0]],device=device,dtype=torch.long)

        batch,_,_,_ = out.shape
        # print(f'batch is {batch}')
        for idx in range(batch):
            parsing_ = out[idx].argmax(0)
            # print(f'out[1] is {out[0].shape}')
            parsing_ = parsing_.type(torch.uint8)
            vis_parsing_anno = parsing_
            # print(f'vis is {vis_parsing_anno}')
            vis_parsing_anno_color = torch.zeros((256, 256, 3),device=device, dtype=torch.long)


            num_of_class = torch.max(vis_parsing_anno)

            for pi in range(0, num_of_class + 1):
                index = torch.where(vis_parsing_anno == pi)# 获得对应分类的的像素坐标
                # print(pi)
                # print(f', index is {index}')
                idx_y = (index[0]).type(torch.long)
                idx_x = (index[1]).type(torch.long)

                # continue
                vis_parsing_anno_color[idx_y,idx_x, :] = part_colors[pi]# 给对应的类别的掩码赋值

                # if pi in[1,2,3,4,5,6,7,8,10,11,12,13,14,17]:
                #     face_mask[idx_y,idx_x] = 0.35

                # vis_parsing_anno_color = vis_parsing_anno_color/255

            vis_parsing_anno_color = vis_parsing_anno_color.permute(2,0,1)/255
            # print(f'vis_parsing_anno_color shape is {vis_parsing_anno_color.shape}')
            fina_img = fg_imgs[idx]*vis_parsing_anno_color +bg_imgs[idx]*(1-vis_parsing_anno_color)
            fina_img = resize_128(fina_img)
            fina_img = fina_img.unsqueeze(0)
            if idx == 0:
                fina_imgs = fina_img
            else:
                fina_imgs = torch.cat((fina_imgs,fina_img), dim=0)

        
    
    return fina_imgs


                
def get_fixed_imgs_temp(fg_imgs, model_path, device):

    """
    bg_imgs: 接收
    """
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net = net.to(device)

    # print('model : {}'.format(model_path))
    net.load_state_dict(torch.load(model_path))
    net.eval()
    trans = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        
    ])
    with torch.no_grad():
        #取背景为mask
        img = trans(fg_imgs)
        # print(f'img shape is {img.shape}')
        out = net(img)[0]
        # print(f'out shape is {out.shape}, out type is {type(out)}')
        # print(f'out shape is {out.squeeze(0).argmax(0).shape}, out type is {type(out)}')
        # idx += 1
        # parsing_ = parsing_.type_as(torch.uint8)
        # print(f'parsing shape is {parsing_.shape}')
        # print(f'parsing_ is {parsing_}')

        
        #按颜色分割
        part_colors = torch.tensor([[0, 0, 0], [255, 255, 255], [255, 255, 255],
                    [255, 255, 255], [255, 255, 255],
                    [255, 255, 255], [255, 255, 255], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0],
                    [255, 255, 255], [255, 255, 255], [255, 255, 255],
                    [255, 255, 255], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0]],device=device,dtype=torch.long)

        batch,_,_,_ = out.shape
        # print(f'batch is {batch}')
        for idx in range(batch):
            parsing_ = out[idx].argmax(0)
            # print(f'out[1] is {out[0].shape}')
            parsing_ = parsing_.type(torch.uint8)
            vis_parsing_anno = parsing_
            # print(f'vis is {vis_parsing_anno}')
            vis_parsing_anno_color = torch.zeros((256, 256, 3),device=device, dtype=torch.long)


            num_of_class = torch.max(vis_parsing_anno)

            for pi in range(0, num_of_class + 1):
                index = torch.where(vis_parsing_anno == pi)# 获得对应分类的的像素坐标
                # print(pi)
                # print(f', index is {index}')
                idx_y = (index[0]).type(torch.long)
                idx_x = (index[1]).type(torch.long)

                # continue
                vis_parsing_anno_color[idx_y,idx_x, :] = part_colors[pi]# 给对应的类别的掩码赋值

                # if pi in[1,2,3,4,5,6,7,8,10,11,12,13,14,17]:
                #     face_mask[idx_y,idx_x] = 0.35

                # vis_parsing_anno_color = vis_parsing_anno_color/255

            vis_parsing_anno_color = vis_parsing_anno_color.permute(2,0,1)/255    
    
        
            if idx == 0:
                out_list = vis_parsing_anno_color.unsqueeze(0)
            else:
                out_list = torch.concat((out_list, vis_parsing_anno_color.unsqueeze(0)), dim=0)
        return out_list


if __name__ == "__main__":
    img_size = 256 # 推理分辨率设置
    model_path = "/home/yl/lk/code/faceparsing-master/fp_256.pth" # 模型路径
    image_path = "/home/yl/lk/code/ID-dise4/fake/small_image/0/"
    bg_path = '/home/yl/lk/code/ID-dise4/out/test_INN8/orin_imgs/'
    fg_path = '/home/yl/lk/code/ID-dise4/out/test_INN8/anonymous_fir_imgs/'
    out_path = '/home/yl/lk/code/ID-dise4/out/test_INN8/anonymous_fir_degh_imgs/'

    for test_idx in range(4):
        raw_img = cv2.imread(bg_path+str(test_idx)+'.png')
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_img = cv2.resize(raw_img,(256,256),interpolation=cv2.INTER_CUBIC)
        raw_img = totensor(raw_img)
        bg_img = raw_img.unsqueeze(0).to('cuda:0')    


        manual_img = cv2.imread(fg_path+str(test_idx)+'.png')
        manual_img = cv2.cvtColor(manual_img, cv2.COLOR_BGR2RGB)
        manual_img = cv2.resize(manual_img,(256,256),interpolation=cv2.INTER_CUBIC)
        manual_img = totensor(manual_img)
        fg_img = manual_img.unsqueeze(0).to('cuda:0')




        out_img = get_fixed_imgs(bg_img,fg_img, model_path, device='cuda:0')

        torchvision.utils.save_image(out_img,out_path+str(test_idx)+'.png')
            # inference(img_size = img_size, image_path=image_path, model_path=model_path)





