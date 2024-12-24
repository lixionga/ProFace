# -*- coding:utf-8 -*-
"""
作者：cyd
日期：2024年11月27日
"""
import numpy as np
import cv2

def crop_face_from_image(image, model, image_size=(380, 380)):
    """
    从输入的图像中裁剪出人脸区域。

    参数:
    image (numpy.ndarray): 输入的图像数据，格式为(height, width, channels)。
    model: 用于预测人脸信息的模型，需包含predict_jsons方法来获取人脸的边界框信息。
    image_size (tuple): 裁剪后图像的目标尺寸，默认为(380, 380)。

    返回:
    list: 包含裁剪后人脸图像的列表，每个元素是裁剪后的人脸图像数据（格式为(channels, height, width)）。
    """
    # 将图像颜色空间转换为RGB（假设模型输入要求为RGB格式）
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用模型预测人脸信息
    faces = model.predict_jsons(frame)
    if len(faces) == 0:
        print('No face is detected')
        return []

    croppedfaces = []
    for face_idx in range(len(faces)):
        x0, y0, x1, y1 = faces[face_idx]['bbox']
        bbox = np.array([[x0, y0], [x1, y1]])
        croppedface = crop_face(frame, None, bbox, False, crop_by_bbox=True, only_img=True, phase='test')
        croppedface = cv2.resize(croppedface, dsize=image_size).transpose((2, 0, 1))
        croppedfaces.append(croppedface)

    return croppedfaces


def crop_face(img, landmark=None, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False,
              phase='train'):
    """
    内部函数，用于根据人脸关键点或边界框信息裁剪图像中的人脸区域。

    参数:
    （此处参数说明与原prepocess文件中的crop_face函数相同，不再赘述）

    返回:
    （此处返回值说明与原prepocess文件中的crop_face函数相同，不再赘述）
    """
    assert phase in ['train', 'val', 'test']

    H, W = len(img), len(img[0])
    assert landmark is not None or bbox is not None

    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 4  # 0#np.random.rand()*(w/8)
        w1_margin = w / 4
        h0_margin = h / 4  # 0#np.random.rand()*(h/5)
        h1_margin = h / 4
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 8  # 0#np.random.rand()*(w/8)
        w1_margin = w / 8
        h0_margin = h / 2  # 0#np.random.rand()*(h/5)
        h1_margin = h / 5

    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        w0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        w1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        h0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        h1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p - x0_new, q - y0_new]
    else:
        landmark_cropped = None
    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        for i, (p, q) in enumerate(bbox):
            bbox_cropped[i] = [p - x0_new, q - y0_new]
    else:
        bbox_cropped = None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped, landmark_cropped, bbox_cropped, (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1), y0_new, y1_new, \
               x0_new, x1_new
    else:
        return img_cropped, landmark_cropped, bbox_cropped, (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1)

if __name__ == '__main__':

    crop_face_from_image()