# -*- coding:utf-8 -*-
"""
作者：cyd
日期：2024年11月27日
"""
import cv2
import dlib
from imutils import face_utils
import os
import numpy as np


def facecrop(org_image_path, save_path):
    """
    函数功能：对给定的单张图像进行人脸剪裁，保存剪裁后的人脸图像以及人脸关键点信息。

    参数：
    org_image_path: 原始图像的路径，字符串类型。
    save_path: 保存结果（剪裁后的人脸图像和关键点信息）的路径，字符串类型。
    """
    # 读取原始图像
    frame_org = cv2.imread(org_image_path)
    if frame_org is None:
        print(f'读取图像 {org_image_path} 出错')
        return

    # 将图像颜色空间转换为RGB（因为后续dlib相关处理可能要求此格式）
    frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)

    # 获取人脸检测器和关键点预测器
    face_detector = dlib.get_frontal_face_detector()
    predictor_path ='weights/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    # 进行人脸检测
    faces = face_detector(frame, 1)
    if len(faces) == 0:
        print(f'在图像 {org_image_path} 中未检测到人脸')
        return

    # 用于记录每个人脸的面积以及对应的关键点
    size_list = []
    landmarks_list = []
    for face_idx in range(len(faces)):
        # 预测人脸关键点
        landmark = face_predictor(frame, faces[face_idx])
        landmark = face_utils.shape_to_np(landmark)
        x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
        x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
        face_s = (x1 - x0) * (y1 - y0)
        size_list.append(face_s)
        landmarks_list.append(landmark)

    # 根据人脸面积大小对关键点数据进行排序（降序），取面积最大的人脸
    landmarks = np.concatenate(landmarks_list).reshape((len(size_list),) + landmark.shape)
    landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

    # 创建保存人脸图像的目录（如果不存在）
    save_image_path = os.path.join(save_path, 'cropped_faces')
    os.makedirs(save_image_path, exist_ok=True)
    image_name = os.path.basename(org_image_path).split('.')[0] + '_cropped.png'
    image_path = os.path.join(save_image_path, image_name)

    # 获取人脸区域的坐标，进行图像剪裁
    x0, y0 = int(landmarks[:, 0].min()), int(landmarks[:, 1].min())
    x1, y1 = int(landmarks[:, 0].max()), int(landmarks[:, 1].max())
    cropped_face = frame_org[y0:y1, x0:x1]
    cv2.imwrite(image_path, cropped_face)

    # 创建保存人脸关键点信息的目录（如果不存在）
    save_land_path = os.path.join(save_path, 'landmarks')
    os.makedirs(save_land_path, exist_ok=True)
    land_path = os.path.join(save_land_path, image_name.replace('.png', '.npy'))
    np.save(land_path, landmarks)


# if __name__ == '__main__':
#     # 示例用法，这里替换为你实际的单张图像路径和想要保存结果的路径
#     org_image_path = 'picture/7.png'
#     save_path = 'dlib_crop'
#     facecrop(org_image_path, save_path)