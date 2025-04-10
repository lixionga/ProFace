# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE
import dlib
import torch
import os
import numpy as np
import cv2
from torch import nn
import sys
import albumentations as alb
import torchvision.transforms.functional as F
from imutils import face_utils
import warnings
from retinaface.pre_trained_models import get_model
from ..utils import blend as B
from ..utils.funcs import IoUfrom2bboxes, crop_face, RandomDownScale

warnings.filterwarnings('ignore')

import FaceSecurity.BBW.config.cfg as c

device = torch.device(c.device if torch.cuda.is_available() else "cpu")

if os.path.isfile('/app/src/utils/library/bi_online_generation.py'):
    sys.path.append('/app/src/utils/library/')
    print('exist library')
    exist_bi = True
else:
    exist_bi = False


class SBI(nn.Module):
    def __init__(self, image_size=224):
        super().__init__()
        self.image_size = (image_size, image_size)
        # Load face detector and shape predictor
        self.face_detector = dlib.get_frontal_face_detector()
        predictor_path = './network/distortions/deepfakes/selfblended/SelfBlendedImage/src/preprocess/shape_predictor_81_face_landmarks.dat'
        self.face_predictor = dlib.shape_predictor(predictor_path)

        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()

        self.model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
        self.model.eval()

    def forward(self, input_image_tensor, landmark):
        # Convert tensor to numpy array
        img_np = np.array(F.to_pil_image(input_image_tensor))
        landmark = np.array(landmark)

        img_r, img_f, mask_f = self.self_blending(img_np.copy(), landmark.copy())

        return img_f

    def retina_landmarks(self, image):
        # Perform face detection
        faces = self.model.predict_jsons(image)
        if len(faces) == 0:
            raise ValueError
            # print(f'retina_landmarks,error in')
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
        return landmarks

    def get_landmark(self, image):
        # Initialize landmarks list
        landmarks = []
        size_list = []

        # Detect faces
        faces = self.face_detector(image, 1)

        if len(faces) == 0:
            raise ValueError

        for face_idx in range(len(faces)):
            landmark = self.face_predictor(image, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

        return landmarks

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),

                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                       val_shift_limit=(-0.3, 0.3), p=1),

                alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1)

            ], p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),

                alb.Blur(blur_limit=5, always_apply=True)
            ], p=1),

        ], p=1.)

    def get_transforms(self):
        return alb.Compose([

            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),

            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                   val_shift_limit=(-0.3, 0.3), p=0.3),

            alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),

            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),

        ],
            additional_targets={f'image1': 'image'},
            p=1.)

    def randaffine(self, img, mask):
        f = alb.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    def self_blending(self, img, landmark):
        H, W = len(img), len(img[0])
        if np.random.rand() < 0.25:
            landmark = landmark[:68]

        # if exist_bi:
        #     # logging.disable(logging.FATAL)
        #     mask = random_get_hull(landmark, img)[:, :, 0]
        #     # logging.disable(logging.NOTSET)

        mask = np.zeros_like(img[:, :, 0])
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        source = img.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source, mask)

        img_blended, mask = B.dynamic_blend(source, img, mask)

        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img, img_blended, mask

    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def hflip(self, img, mask=None, landmark=None, bbox=None):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        bbox = bbox.copy()

        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]

            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise ValueError
            landmark_new[:, 0] = W - landmark_new[:, 0]

        else:
            landmark_new = None

        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W - bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W - bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W - bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W - bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W - bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W - bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None

        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        img = img[:, ::-1].copy()
        return img, mask, landmark_new, bbox_new

    def collate_fn(self, batch):
        img_f, img_r = zip(*batch)
        data = {}
        data['img'] = torch.cat([torch.tensor(img_r).float(), torch.tensor(img_f).float()], 0)
        data['label'] = torch.tensor([0] * len(img_r) + [1] * len(img_f))
        return data

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
