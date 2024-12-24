import torch
from torch.utils.data import Dataset,IterableDataset
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import albumentations as alb

import warnings
warnings.filterwarnings('ignore')

import logging

if os.path.isfile('src/utils/library/bi_online_generation.py'):
    sys.path.append('src/utils/library/')
    exist_bi=True
else:
    exist_bi=False


class Dataset(Dataset):
    def __init__(self,phase='train', image_size=224, n_frames=8, fake_target=False, fake_local=False, random=0.5, mask_setting = None):
        
        assert phase in ['train','val','test']
        
        self.target_domain = fake_target
        self.fake_local = fake_local
        
        if not self.target_domain:
            image_list, label_list = init_ff(phase, 'frame', n_frames=n_frames)
        else:
            image_list, label_list, fake_image_list, fake_label_list = init_ff_all(phase, 'frame', n_frames=n_frames)
            
        path_lm='/landmarks/' 
        label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
        image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
        
        if self.target_domain:
            fake_image_list=[fake_image_list[i] for i in range(len(fake_image_list)) if os.path.isfile(fake_image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(fake_image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
            self.fake_image_list = fake_image_list

        self.image_list=image_list

        self.image_size=(image_size,image_size)
        self.phase=phase
        self.n_frames=n_frames

        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.random = random
        self.mask_setting = mask_setting
        
        self.path_lm = path_lm
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        flag=True
        while flag:
            try:
                filename=self.image_list[idx]
                img=np.array(Image.open(filename))
                
                if self.target_domain:
                    filename_fake=self.fake_image_list[int(idx * 1.998 + random.choice([0,1]))]
                    img_fake=np.array(Image.open(filename_fake))
                    
                landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
                bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
                bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
                iou_max=-1
                
                for i in range(len(bboxes)):
                    iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
                    if iou_max<iou:
                        bbox=bboxes[i]
                        iou_max=iou

                landmark=self.reorder_landmark(landmark) # 68/81 landmarks
                
                if self.phase=='train':
                    if np.random.rand() < 0.5:
                        img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)

                img, landmark, bbox, __ = crop_face(img, landmark, bbox, margin=True, crop_by_bbox=False) 

                if self.target_domain:
                    landmark_fake=np.load(filename_fake.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
                    bbox_lm_fake=np.array([landmark_fake[:,0].min(),landmark_fake[:,1].min(),landmark_fake[:,0].max(),landmark_fake[:,1].max()])
                    bboxes_fake=np.load(filename_fake.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
                    iou_max_fake=-1
                    
                    for i in range(len(bboxes_fake)):
                        iou_fake=IoUfrom2bboxes(bbox_lm_fake,bboxes_fake[i].flatten())
                        if iou_max_fake<iou_fake:
                            bbox_fake=bboxes_fake[i]
                            iou_max_fake=iou_fake

                    landmark_fake=self.reorder_landmark(landmark_fake) # 68/81 landmarks
                    
                    if self.phase=='train':
                        if np.random.rand() < 0.5:
                            img_fake,_,landmark_fake,bbox_fake=self.hflip(img_fake,None,landmark_fake,bbox_fake)

                    img_fake, landmark_fake, bbox_fake, __ = crop_face(img_fake, landmark_fake, bbox_fake, margin=True, crop_by_bbox=False)

                
                img_r, blend_imgs = self.self_blending(img.copy(), landmark.copy(), self.fake_local, setting=self.mask_setting)
                
                img_f_g, mask_g, source = blend_imgs[0]
                
                if self.fake_local:
                    img_f_l, mask_l = blend_imgs[1]
                
                if self.phase=='train': 
                    if self.target_domain:
                        if self.fake_local:
                            transformed=self.transforms(image=img_f_g.astype('uint8'), image1=img_r.astype('uint8'), image2=img_fake.astype('uint8'), image3=img_f_l.astype('uint8'))
                            img_f_g=transformed['image']
                            img_r=transformed['image1']
                            img_fake=transformed['image2']
                            img_f_l=transformed['image3']
                        else:
                            transformed=self.transforms(image=img_f_g.astype('uint8'), image1=img_r.astype('uint8'), image2=source.astype('uint8'), image3=img_fake.astype('uint8'))
                            img_f_g=transformed['image']
                            img_r=transformed['image1']
                            source=transformed['image2']
                            img_fake=transformed['image3']

                    else:
                        if self.fake_local:
                            transformed=self.transforms(image=img_f_g.astype('uint8'), image1=img_r.astype('uint8'), image2=img_f_l.astype('uint8'))
                            img_f_g=transformed['image']
                            img_r=transformed['image1']
                            img_f_l=transformed['image2']
                        else:
                            transformed=self.transforms(image=img_f_g.astype('uint8'), image1=img_r.astype('uint8'), image2=source.astype('uint8'))
                            img_f_g=transformed['image']
                            img_r=transformed['image1']
                            source=transformed['image2']
                                 
                img_f_g,_,__,___,y0_new,y1_new,x0_new,x1_new = crop_face(img_f_g,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
                
                if self.target_domain:
                    img_fake,_,__,___,____,_____,______,_______ = crop_face(img_fake,landmark_fake,bbox_fake,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
                    
                img_r = img_r[y0_new:y1_new, x0_new:x1_new]
                mask_g = mask_g[y0_new:y1_new, x0_new:x1_new]
                source = source[y0_new:y1_new, x0_new:x1_new]
                
                del _
                del __
                del ___

                img_f_g = cv2.resize(img_f_g, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32')/255
                img_r = cv2.resize(img_r, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32')/255
                mask_g = cv2.resize(mask_g, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32')
                
                source = cv2.resize(source, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32')/255
                  
                
                img_f_g = img_f_g.transpose((2,0,1))
                img_r = img_r.transpose((2,0,1))

                
                mask_g = mask_g[None, :, :]

                source = source.transpose((2,0,1))
                
                if self.fake_local:
                    img_f_l = img_f_l[y0_new:y1_new,x0_new:x1_new]
                    mask_l = mask_l[y0_new:y1_new,x0_new:x1_new]
                    img_f_l = cv2.resize(img_f_l,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                    mask_l = cv2.resize(mask_l,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255 
                    img_f_l=img_f_l.transpose((2,0,1))
                    mask_l=mask_l[None,:,:] / (np.max(mask_l)+1e-9)
                    
                if self.target_domain:
                    img_fake = cv2.resize(img_fake,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                    img_fake = img_fake.transpose((2,0,1))
                
                flag = False
    
            except Exception as e:
                idx=torch.randint(low=0,high=len(self),size=(1,)).item()
        
        if self.target_domain:
            if self.fake_local:
                return img_f_g, img_r, mask_g, img_fake, img_f_l, mask_l
            else:
                return img_f_g, img_r, mask_g, source, img_fake
        else:
            if self.fake_local:
                return img_f_g, img_r, mask_g, img_f_l, mask_l
            else:
                return img_f_g, img_r, mask_g, source
        
        
    def get_source_transforms(self):
        return alb.Compose([
                alb.Compose([
                        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                        alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                    ],p=1),
    
                alb.OneOf([
                    RandomDownScale(p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1), 
                ],p=1),
                
            ], p=1.)

        
    def get_transforms(self):
        return alb.Compose([
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
        ], 
        additional_targets={f'image1': 'image', 'image2': 'image', 'image3': 'image'},
        p=1.)

    
    def get_distorts(self):
        return alb.Compose([
            alb.GaussNoise((10,50), p=0.5),
            alb.augmentations.transforms.PixelDropout(p=0.5)
        ], 
        additional_targets={f'image1': 'image'},
        p=0.1)
        
    
    def get_distorts_v2(self, img, threshold):
        if random.random() < threshold:
            f = alb.Affine(
                translate_percent={'x':(-0.005,0.005),'y':(-0.005,0.005)},
                interpolation=cv2.BORDER_REPLICATE,
                p=1)
            
            transformed = f(image = img)
            ratio = random.uniform(0.3, 0.7)
            img_t = transformed['image']
            img = ratio * img + (1 - ratio) * img_t
            
        return img
        
        
    def randaffine(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']
        
        mask=transformed['mask'] 
        transformed=g(image=img,mask=mask)
        mask=transformed['mask'] 
        
        return img, mask

    def randaffine_local(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']
        
        mask=transformed['mask'] 
        transformed=g(image=img,mask=mask)
        mask=transformed['mask'] 
        
        return img,mask
    
    def self_blending(self, img, landmark, local=False, setting=None):
        H,W = len(img),len(img[0])
        
        if np.random.rand()<0.25:
            landmark=landmark[:68]
            
        if exist_bi:
            if local:
                mask_local = random_get_local_hull(landmark, img)
                
            if setting is None:
                mask = random_get_hull(landmark, img)[:,:,0] 
            else:
                mask = target_get_hull(landmark, img, setting)[:,:,0] 
        else:
            mask=np.zeros_like(img[:,:,0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        source = img.copy()
        
        if np.random.rand() < 0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source, mask)
        img_blended, mask = B.dynamic_blend(source, img, mask) 

        if local:
            source = img.copy()
            source, mask_local = self.randaffine_local(source, mask_local)
            img_blended_local, mask_local = B.dynamic_blend_local(source, img, mask_local)
        
            img_blended_local = img_blended_local.astype(np.uint8)
            img_blended = img_blended.astype(np.uint8)
            
        img = img.astype(np.uint8)
        
        if local:
            return img, [(img_blended, mask), (img_blended_local, mask_local)]
        else:
            return img, [(img_blended, mask, source)]
        
        
    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark

    def hflip(self,img,mask=None,landmark=None,bbox=None):
        H,W=img.shape[:2]

        if landmark is not None:
            landmark=landmark.copy()
            landmark_new=np.zeros_like(landmark)

            landmark_new[:17]=landmark[:17][::-1]
            landmark_new[17:27]=landmark[17:27][::-1]

            landmark_new[27:31]=landmark[27:31]
            landmark_new[31:36]=landmark[31:36][::-1]

            landmark_new[36:40]=landmark[42:46][::-1]
            landmark_new[40:42]=landmark[46:48][::-1]

            landmark_new[42:46]=landmark[36:40][::-1]
            landmark_new[46:48]=landmark[40:42][::-1]

            landmark_new[48:55]=landmark[48:55][::-1]
            landmark_new[55:60]=landmark[55:60][::-1]

            landmark_new[60:65]=landmark[60:65][::-1]
            landmark_new[65:68]=landmark[65:68][::-1]
            
            if len(landmark)==68:
                pass
            elif len(landmark)==81:
                landmark_new[68:81]=landmark[68:81][::-1]
            else:
                raise NotImplementedError
            
            landmark_new[:,0]=W-landmark_new[:,0]
            
        else:
            landmark_new=None

        if bbox is not None:
            bbox=bbox.copy()
            bbox_new=np.zeros_like(bbox)
            bbox_new[0,0]=bbox[1,0]
            bbox_new[1,0]=bbox[0,0]
            bbox_new[:,0]=W-bbox_new[:,0]
            bbox_new[:,1]=bbox[:,1].copy()
            if len(bbox)>2:
                bbox_new[2,0]=W-bbox[3,0]
                bbox_new[2,1]=bbox[3,1]
                bbox_new[3,0]=W-bbox[2,0]
                bbox_new[3,1]=bbox[2,1]
                bbox_new[4,0]=W-bbox[4,0]
                bbox_new[4,1]=bbox[4,1]
                bbox_new[5,0]=W-bbox[6,0]
                bbox_new[5,1]=bbox[6,1]
                bbox_new[6,0]=W-bbox[5,0]
                bbox_new[6,1]=bbox[5,1]
        else:
            bbox_new=None

        if mask is not None:
            mask=mask[:,::-1]
        else:
            mask=None
        img=img[:,::-1].copy()
        return img,mask,landmark_new,bbox_new
    
    def collate_fn(self, batch):
        data={}
        if not self.target_domain:
            if self.fake_local:
                img_f, img_r, mask_g, img_f_l, mask_l = zip(*batch)
            else:
                img_f, img_r, mask_g, source = zip(*batch)
        else:
            if self.fake_local:
                img_f, img_r, mask_g, img_fake, img_f_l, mask_l = zip(*batch)
            else:
                img_f, img_r, mask_g, source, img_fake = zip(*batch)
        
        data['img'] = torch.cat([torch.tensor(img_r).float(), torch.tensor(img_f).float()],0)
        data['label'] = torch.tensor([0]*len(img_r)+[1]*len(img_f))
        data['mask'] = torch.cat([torch.zeros_like(torch.tensor(mask_g).float()).float(), torch.tensor(mask_g).float()],0)
        data['source'] = torch.tensor(source).float()
        
        if self.fake_local:
            data['local_fake']=torch.tensor(img_f_l).float()
            data['local_fake_mask']=torch.tensor(mask_l).float()
        
        if self.target_domain:
            data['img_fake']=torch.tensor(img_fake).float()
        
        return data
        

    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__=='__main__':
    import blend as B
    from initialize import *
    from funcs import IoUfrom2bboxes,crop_face,RandomDownScale
 
    if exist_bi:
        from library.bi_online_generation import random_get_hull, random_get_local_hull, get_local_hull, target_get_hull
  
    seed=10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
    image_dataset=Dataset(phase='train',image_size=256)
    batch_size=2
    dataloader = torch.utils.data.DataLoader(image_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=image_dataset.collate_fn,
                    num_workers=0,
                    worker_init_fn=image_dataset.worker_init_fn
                    )
 
    data_iter=iter(dataloader)
    data=next(data_iter)
 
else:
    from utils import blend as B
    from .initialize import *
    from .funcs import IoUfrom2bboxes,crop_face,RandomDownScale
    if exist_bi:
        from utils.library.bi_online_generation import random_get_hull, random_get_local_hull, get_local_hull, target_get_hull