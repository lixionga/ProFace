import json
import numpy as np
import albumentations as alb
import cv2
import torch
import random
import copy
import random
import torch.nn as nn
from utils.blend import get_blend_mask_simple

def load_json(path):
    d = {}
    with open(path, mode="r") as f:
        d = json.load(f)
    return d


def Bbox_generation(mask, cam, gpu):
    mask = mask.cpu().squeeze(1).numpy()
    mask_boxes, cam_boxes = [], []
    cam_binary = copy.deepcopy(cam)
    
    for i in range(mask.shape[0]):
        cam_binary[i][cam_binary[i] > cam_binary[i].mean()] = 1
        cam_binary[i][cam_binary[i] < 1] = 0
        cam_binary[i] = cam_binary[i] * 255

        temp = np.array(cam_binary[i], dtype=np.uint8)
        temp = temp[:,:,None]

        cam_contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        size_cam = 0

        if len(cam_contours) > 0:
            for contour in cam_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > size_cam:
                    size_cam = w * h
                    bounding_boxes_cam = [x, y, x + w, y + h]
        else:
            bounding_boxes_cam = [0, 0, 0, 0]
            
        cam_boxes.append(bounding_boxes_cam)

        mask[i][mask[i] > mask[i].mean()] = 1
        mask[i][mask[i] < 1] = 0
        mask[i] = mask[i] * 255

        temp = np.array(mask[i], dtype=np.uint8)
        temp = temp[:,:,None]
        mask_contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        size_mask = 0
        
        if len(mask_contours) > 0:
            for contour in mask_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > size_mask:
                    size_mask = w * h
                    bounding_boxes_mask = [x, y, x + w, y + h]
        else:
            bounding_boxes_mask = [0, 0, 0, 0]
                    
        mask_boxes.append(bounding_boxes_mask)
        
    cam_boxes = torch.Tensor(cam_boxes).to(gpu)
    mask_boxes = torch.Tensor(mask_boxes).to(gpu)
    
    return mask_boxes, cam_boxes
    

def IoUfrom2bboxes(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def IoU(boxesA, boxesB):
    a_left_top = boxesA[:, :2]
    a_right_bottom = boxesA[:, 2:]
    b_left_top = boxesB[:, :2]
    b_right_bottom = boxesB[:, 2:]

    intersection_left_top = torch.max(a_left_top, b_left_top)
    intersection_right_bottom = torch.min(a_right_bottom, b_right_bottom)

    intersection_area = torch.prod(torch.clamp(intersection_right_bottom - intersection_left_top, min=0), dim=-1)

    a_area = torch.prod(a_right_bottom - a_left_top, dim=-1)
    b_area = torch.prod(b_right_bottom - b_left_top, dim=-1)


    iou = intersection_area / (a_area + b_area - intersection_area)
    
    return iou


def shape_refinemnet(img, cam, mask, idx, model, ref_outputs, threshold = 0.5, random_thresh = 0.5):
    weight = torch.zeros_like(img)
    weight = weight[:,0]
    b = img.shape[0]
    gpu = model.device
    
    for i in idx:
        r_img = copy.deepcopy(img[i - b//2])
        f_img = copy.deepcopy(img[i])
        
        cam_i = copy.deepcopy(cam[i])
        mx = np.max(cam_i)
            
        if mx != 0:
            ratio = 1/mx
        else:
            ratio = 1
            
        cam_i = cam_i[None] * ratio
            
        mask_i = mask[i].cpu().numpy()
        mask_t = np.ones_like(cam_i)
        mask_t[mask_i == 0] = 0 
        
        if i < b//2:
            ref = 0
        else:
            ref = 1

        m = np.mean(cam_i[cam_i > 0])
        delta_i = np.zeros_like(cam_i)
        delta_i[cam_i > m] = 1
        
        idx1 = mask_i > 0 
        idx2 = delta_i == 1 
        common_condition= np.logical_and(idx1, idx2)
        common_idx = np.where(common_condition)
        
        delta = np.zeros_like(mask_t)
        delta[common_idx] = 1
        
        NUM_delta = np.sum(delta == 1)
        NUM_mask = np.sum(mask_t == 1) + 1e-12 
        
        pred = ref_outputs[i].softmax(dim=-1)[ref].cpu().numpy()   

        if (NUM_delta / NUM_mask) < threshold and (pred > 0.5) and (i >= b//2) and (NUM_mask > 1):
            if np.random.choice([0, 1], p=np.array([1-pred, pred])):
                r_img = img[i - b//2].clone()
                f_img = img[i].clone()
                
                delta_i2 = copy.deepcopy(mask_t)
                delta_i2[common_idx] = 0
                delta_i2 = delta_i2[0]

                delta_i_mask = get_blend_mask_simple(delta_i2)[:,:,0]
                delta_i_mask = torch.tensor(delta_i_mask).to(gpu)[None]
    
                mask[i] = mask[i] - mask[i]
                mask[i - b//2] = mask[i - b//2] - mask[i - b//2]
                mask[i] = mask[i] + delta_i_mask.clone()
                mask[i - b//2] = mask[i - b//2] + delta_i_mask.clone()
                
                if torch.sum(mask[i]) == 0:
                    print('Warning!')
                    
                delta_i_mask = delta_i_mask.repeat(3, 1, 1)
                img[i] = (1 - delta_i_mask) * r_img + delta_i_mask * f_img
    
    return img, mask


def magnitude_refinement(img, source, mask, target, model=None, criterion=nn.CrossEntropyLoss(), EPS = 0.01, random_thresh = 0.5):
    b = img.shape[0]
    background = img[:b//2]
    blending_mask = mask[:b//2]
    fake_img2 = img[b//2:]

    source = source.clone().detach().to(source.device)
    background = background.clone().detach().to(background.device)
    fake_img = fake_img2.clone().detach().to(fake_img2.device)
    blending_mask_tmp = blending_mask.clone().detach().to(blending_mask.device)
    fake_label = target[b//2:].clone().detach().to(target.device)
    
    delta = fake_img - (source * blending_mask_tmp + background * (1 - blending_mask_tmp))
    
    if model is not None:
        fake_img.requires_grad = True
        loss = criterion(model(fake_img), fake_label)
        img_grad = torch.autograd.grad(loss, fake_img)[0]
        blending_mask_grad = img_grad * (source - background)
        
        if random.random() > random_thresh:
            EPS_REAL = EPS
        else:
            EPS_REAL = 0
        
        label = torch.sum(blending_mask_grad > 0, dim=(1,2,3)) > torch.sum(blending_mask_grad < 0, dim=(1,2,3))
        label = 2 * label - 1
        label = label[:, None, None, None]
        
        blending_mask_tmp = blending_mask_tmp * (1 + EPS_REAL * label)
        blending_mask_tmp = torch.clamp(blending_mask_tmp, 0, 1)
        
    forgery = (source * blending_mask_tmp + background * (1 - blending_mask_tmp)) + delta
    forgery = torch.clamp(forgery, 0, 1)
    forgery = forgery.detach()
    
    img_new = torch.cat((background, forgery), dim=0)

    return img_new.contiguous()


def crop_face(img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
    assert phase in ['train','val','test']

    #crop face------------------------------------------
    H,W=len(img),len(img[0])

    assert landmark is not None or bbox is not None

    H,W=len(img),len(img[0])
    
    if crop_by_bbox:
        x0,y0=bbox[0]
        x1,y1=bbox[1]
        w=x1-x0
        h=y1-y0
        w0_margin=w/4
        w1_margin=w/4
        h0_margin=h/4
        h1_margin=h/4
    else:
        x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
        x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
        w=x1-x0
        h=y1-y0
        w0_margin=w/8
        w1_margin=w/8
        h0_margin=h/2
        h1_margin=h/5

    if margin: 
        w0_margin*=4
        w1_margin*=4
        h0_margin*=2
        h1_margin*=2
    elif phase=='train':
        w0_margin*=(np.random.rand()*0.6+0.2)
        w1_margin*=(np.random.rand()*0.6+0.2)
        h0_margin*=(np.random.rand()*0.6+0.2)
        h1_margin*=(np.random.rand()*0.6+0.2)
    else:
        w0_margin*=0.5
        w1_margin*=0.5
        h0_margin*=0.5
        h1_margin*=0.5
    	
    y0_new=max(0,int(y0-h0_margin))
    y1_new=min(H,int(y1+h1_margin)+1)
    x0_new=max(0,int(x0-w0_margin))
    x1_new=min(W,int(x1+w1_margin)+1)
 
    img_cropped=img[y0_new:y1_new,x0_new:x1_new]
 
    if landmark is not None:
        landmark_cropped=np.zeros_like(landmark)
        for i,(p,q) in enumerate(landmark):
            landmark_cropped[i]=[p-x0_new,q-y0_new]
    else:
        landmark_cropped=None
    if bbox is not None:
        bbox_cropped=np.zeros_like(bbox)
        for i,(p,q) in enumerate(bbox):
            bbox_cropped[i]=[p-x0_new,q-y0_new]
    else:
        bbox_cropped=None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
    else:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self,img,**params):
        return self.randomdownscale(img)

    def randomdownscale(self,img):
        keep_ratio=True
        keep_input_shape=True
        H,W,C=img.shape
        ratio_list=[2,4]
        r=ratio_list[np.random.randint(len(ratio_list))] # 2/4
        img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

        return img_ds
    