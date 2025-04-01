# -*- coding:utf-8 -*-
"""
作者：cyd
日期：2024年11月28日
"""
import os
import torch
import numpy as np
import os
import random

from forgery_model import Detector
import argparse

#from retinaface.pre_trained_models import get_model
import cv2
import heatmap_generator
from dlib_crop_face import facecrop
from preprocess import extract_face
import warnings
from prepare_data import LandmarkModel
from align_face import align_img

warnings.filterwarnings('ignore')

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

"""
    主函数，用于执行整个检测和生成热图相关的流程
    :param args: 命令行参数对象，包含了各种配置参数
 """
def main(args):


    model=Detector()
    model=model.to(device)
     # 加载模型权重的操作
    cnn_sd=torch.load(args.weight_name,map_location=torch.device(device))["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    # 读取图像
    frame = cv2.imread(args.input_image)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    app = LandmarkModel(name='landmarks')
    app.prepare(ctx_id=0,det_thresh=0.5,det_size=(224,224),mode='None')

    landmark,_ = app.get(frame)
    face,_ = align_img(frame,landmark)


     # 加载人脸检测模型并设为评估模式
    #  face_detector = get_model_from_local("resnet50_2020-07-20", max_size=max(frame.shape),device=device,weights_path='weights/retinaface_resnet50_2020-07-20.pth')
    #  face_detector.eval()

    #  face_list=extract_face(frame,face_detector)

    with torch.no_grad():
        img=cv2tensor(face)
        # torchvision.utils.save_image(img, f'test.png', nrow=8, normalize=False, range=(0, 1))
        pred=model(img).softmax(1)[:,1].cpu().data.numpy().tolist()

    print(f'fakeness: {max(pred):.4f}')

    save_crop_path = "/Users/mac/代码/web/forgery/cropped_results"
    # 调用人脸裁剪函数，对图像进行裁剪，传入原始图像路径和保存结果的路径
    # facecrop(args.input_image, save_crop_path)
    # 修改此处的图像路径，使用裁剪后图像的路径，这里假设裁剪后的图像命名规则和原始代码中一致
    # cropped_image_path = os.path.join(save_crop_path, 'cropped_faces',
    #                                   os.path.basename(args.input_image).split('.')[0] + '_cropped.png')
    image_name = os.path.basename(args.input_image).split('.')[0] + '_cropped.png'
    image_path = os.path.join(save_crop_path, image_name)
    print(f"Saving cropped image to {image_path}")
    cv2.imwrite(image_path, face)
    if image_path is not None and os.path.exists(image_path):
        heatmap_generator.generate_cam_image(device, image_path, aug_smooth=args.aug_smooth,
                                             eigen_smooth=args.eigen_smooth,
                                             method=args.method,
                                             output_dir=args.output_dir)
    else:
        print(f"Error: Invalid cropped image path: {image_path}")

if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 判断CUDA是否可用，根据结果设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("CUDA is not available, using CPU instead.")
        device = torch.device('cpu')


    parser = argparse.ArgumentParser()

    # 定义命令行参数，指定模型权重文件的名称，默认值为'weights/FFraw.tar'
    parser.add_argument('-w', dest='weight_name', type=str, default='/Users/mac/代码/web/forgery/weights/EG_FF++(raw).tar')
    # 定义命令行参数，指定输入图像的路径，默认值为'picture/1.png
    parser.add_argument('-input_image', dest='input_image', type=str, default='/Users/mac/代码/web/processed/processed_1739178219_ds.jpg')
    # 定义命令行参数，指定运行模型的设备，默认值为'cpu'，并提供相应帮助信息
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the models, e.g., cpu or cuda')
    # 定义命令行参数，用于指定是否应用增强平滑，是一个布尔型参数，默认不启用，用于控制相关处理逻辑
    parser.add_argument('--aug_smooth', action='store_true', help='Whether to apply augmentation smoothing')
    # 定义命令行参数，用于指定是否应用特征向量平滑，是一个布尔型参数，默认不启用，用于控制相关处理逻辑
    parser.add_argument('--eigen_smooth', action='store_true', help='Whether to apply eigenvector smoothing')
    # 定义命令行参数，指定生成额外图像所使用的方法，默认值为'gradcam++'，具体根据实际的图像生成逻辑来使用
    parser.add_argument('--method', type=str, default='gradcam++',
                        help='Method to use for generating additional images')
    # 定义命令行参数，指定生成图像的输出目录，默认值为'cam_output'
    parser.add_argument('--output_dir', type=str, default='/Users/mac/代码/web/forgery/cam_output',
                        help='Output directory for generated images')
    args = parser.parse_args()

    main(args)
