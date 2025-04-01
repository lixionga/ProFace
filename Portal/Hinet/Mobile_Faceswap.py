from models.mobile_model import FaceSwap,l2_norm
from models.align_face import dealign,align_img
from models.prepare_data import LandmarkModel
from models.util import tesnor2cv,cv2tensor
import torch
import cv2
import numpy as np

def get_id(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2tensor(id_img)
    mean = torch.tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = torch.tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std
    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)
    return id_emb, id_feature

class Mobile_face:
    def __init__(self,id_emb,id_feature):
        mode = 'None'
        
        faceswap_model = FaceSwap()
        weight = torch.load('/Users/mac/代码/web/checkpoints/model.pth')


        app = LandmarkModel(name='landmarks')
        app.prepare(ctx_id=0,det_thresh=0.5,det_size=(224,224),mode=mode)

        faceswap_model.set_model_param(id_emb,id_feature,weight)
        faceswap_model.eval()
        self.faceswap_model = faceswap_model
        self.app = app

    def img_swap(self, img):
        # image = cv2.imread(img)

        landmark,_ = self.app.get(img)
        face,back_matrix = align_img(img,landmark)

        align_image = cv2tensor(face)

        res, mask = self.faceswap_model(align_image)
        res = tesnor2cv(res)
        mask = np.transpose(mask[0].detach().cpu().numpy(), (1, 2, 0))
        out = dealign(res, img, back_matrix, mask)

        return out
    
    def frame_swap(self,frame):

        landmark,_ = self.app.get(frame)
        face,back_matrix = align_img(frame,landmark)
        align_image = cv2tensor(face)
        res,mask = self.faceswap_model(align_image)
        res = tesnor2cv(res)
        mask = np.transpose(mask[0].detach().cpu().numpy(), (1, 2, 0))
        out = dealign(res, frame, back_matrix, mask)
        return out
    
    def GaussianBlur_frame(self,frame):
        landmarks,bboxes = self.app.gets(frame)
        try:
            for bbox in bboxes:
                frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = cv2.GaussianBlur(
                    frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], (99, 99), 15)

            return frame
        except:
            return frame
        
    def Mosaic_frame(self,frame):
        landmarks,bboxes = self.app.gets(frame)
        try:
            for bbox in bboxes:
                x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3])
                face_region = frame[y1:y2, x1:x2]
                scale_factor = 0.08  # 缩小的比例因子
                small = cv2.resize(face_region, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                mosaic = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

                frame[y1:y2, x1:x2] = mosaic
            return frame
        except:
            return frame