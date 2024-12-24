from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
from imutils import face_utils
from retinaface.pre_trained_models import get_model
import torch
import get_model_from_local

def facecrop_image(model, image_path, save_path):
    """
    从给定的图像中裁剪出人脸区域并保存，同时保存人脸关键点信息

    :param model: 预训练的人脸检测模型
    :param image_path: 输入图像的路径
    :param save_path: 保存裁剪后人脸图像和关键点信息的路径
    :return: 裁剪后的人脸图像（以numpy数组形式返回，如果未检测到人脸则返回None）
    """
    # 读取图像
    frame_org = cv2.imread(image_path)
    if frame_org is None:
        tqdm.write(f'Image read error! : {os.path.basename(image_path)}')
        return None
    height, width = frame_org.shape[:-1]
    frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)

    # 使用模型检测人脸
    faces = model.predict_jsons(frame)
    if len(faces) == 0:
        tqdm.write(f'No faces in {os.path.basename(image_path)}')
        return None
    face_s_max = -1
    landmarks = []
    size_list = []
    for face_idx in range(len(faces)):
        x0, y0, x1, y1 = faces[face_idx]['bbox']
        landmark = np.array([[x0, y0], [x1, y1]] + faces[face_idx]['landmarks'])
        face_s = (x1 - x0) * (y1 - y0)
        size_list.append(face_s)
        landmarks.append(landmark)

    landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
    landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

    # 构建保存人脸图像和关键点信息的路径
    save_path_ = save_path + 'images/' + os.path.basename(image_path).replace('.jpg', '/').replace('.png', '/')
    os.makedirs(save_path_, exist_ok=True)
    image_file_name = os.path.basename(image_path).split('.')[0]
    image_path_saved = save_path_ + image_file_name + '_cropped.png'
    land_path = save_path_ + image_file_name
    land_path = land_path.replace('/images', '/landmarks')
    os.makedirs(os.path.dirname(land_path), exist_ok=True)

    # 获取最大人脸的坐标信息用于裁剪，修改此处的坐标获取逻辑
    x0, y0 = np.round(landmarks[0][0, 0].astype(int)), np.round(landmarks[0][0, 1].astype(int))
    x1, y1 = np.round(landmarks[0][1, 0].astype(int)), np.round(landmarks[0][1, 1].astype(int))
    cropped_face = frame_org[y0:y1, x0:x1]

    # 保存裁剪后的人脸图像和关键点信息
    cv2.imwrite(image_path_saved, cropped_face)
    np.save(land_path, landmarks[0])

    return cropped_face

if __name__ == '__main__':
    # 这里假设通过命令行参数传入图像路径和保存路径，简单示例暂不使用argparse，你可按需完善
    image_path = 'pictures_folder/3.png'  # 替换为实际的图像路径
    save_path = 'crop_face/'  # 替换为实际的保存路径

    # 判断cuda是否可用，不可用则使用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 调用新函数从本地加载权重，替换weights_path为实际的本地权重文件路径
    model = get_model_from_local.get_model_from_local("resnet50_2020-07-20", max_size=2048, device=device, weights_path='weights/retinaface_resnet50_2020-07-20.pth')
    model.eval()

    cropped_image = facecrop_image(model, image_path, save_path)
    if cropped_image is not None:
        print("人脸裁剪成功，已保存裁剪后的人脸图像及关键点信息。")
    else:
        print("图像中未检测到人脸，无法进行裁剪操作。")