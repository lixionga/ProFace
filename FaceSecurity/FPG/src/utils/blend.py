import cv2
import numpy as np
import random


def alpha_blend(source,target,mask):
    mask_blured = get_blend_mask(mask)
    img_blended=(mask_blured * source + (1 - mask_blured) * target)
    return img_blended,mask_blured


def dynamic_blend(source, target, mask):
    mask_blured = get_blend_mask(mask)
    blend_list=[0.25, 0.5, 0.75, 1, 1, 1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured*=blend_ratio
    img_blended=(mask_blured * source + (1 - mask_blured) * target)
 
    return img_blended, mask_blured

def dynamic_blend_local(source, target, mask):
    mask_blured = get_blend_mask_consistant(mask)
    
    blend_list=[0.25, 0.5, 0.75, 1, 1, 1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    
    mask_blured *= blend_ratio
    img_blended=(mask_blured * source + (1 - mask_blured) * target)
 
    return img_blended, mask_blured


def get_blend_mask(mask):
    H,W=mask.shape
    size_h=np.random.randint(192,257)
    size_w=np.random.randint(192,257)
    mask=cv2.resize(mask,(size_w,size_h))
    kernel_1=random.randrange(5,26,2)
    kernel_1=(kernel_1,kernel_1)
    kernel_2=random.randrange(5,26,2)
    kernel_2=(kernel_2,kernel_2)
    
    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured[mask_blured<1]=0
    
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured = cv2.resize(mask_blured,(W,H))
    
    return mask_blured.reshape((mask_blured.shape+(1,)))


def get_blend_mask_consistant(mask):
    H,W=mask.shape
    kernel_0=random.randrange(5,7,5)
    kernel_0=(kernel_0,kernel_0)
    kernel_1=random.randrange(5,7,2)
    kernel_1=(kernel_1,kernel_1)
    kernel_2=random.randrange(5,7,2)
    kernel_2=(kernel_2,kernel_2)
    
    mask = cv2.dilate(mask, kernel_0, iterations=10)
 
    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured[mask_blured<1]=0
    
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured = cv2.resize(mask_blured,(W,H))
 
    return mask_blured.reshape((mask_blured.shape+(1,)))


def get_blend_mask_simple(mask):
    H,W=mask.shape
    
    kernel_0=random.randrange(10,16,2)
    kernel_0=(kernel_0,kernel_0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel_0)
    
    size_h=np.random.randint(192,257)
    size_w=np.random.randint(192,257)
    mask=cv2.resize(mask,(size_w,size_h))
    
    kernel_1=random.randrange(17,26,2)
    kernel_1=(kernel_1,kernel_1)
    kernel_2=random.randrange(17,26,2)
    kernel_2=(kernel_2,kernel_2)
    
    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured[mask_blured<1]=0
    
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(15, 46))
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured = cv2.resize(mask_blured,(W,H))

    
    return mask_blured.reshape((mask_blured.shape+(1,)))


def get_alpha_blend_mask(mask):
    kernel_list=[(11,11),(9,9),(7,7),(5,5),(3,3)]
    blend_list=[0.25,0.5,0.75]
    kernel_idxs=random.choices(range(len(kernel_list)), k=2)
    blend_ratio = blend_list[random.sample(range(len(blend_list)), 1)[0]]
    mask_blured = cv2.GaussianBlur(mask, kernel_list[0], 0)
    mask_blured[mask_blured<mask_blured.max()]=0
    mask_blured[mask_blured>0]=1
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_list[kernel_idxs[1]], 0)
    mask_blured = mask_blured/(mask_blured.max())
    
    return mask_blured.reshape((mask_blured.shape+(1,)))