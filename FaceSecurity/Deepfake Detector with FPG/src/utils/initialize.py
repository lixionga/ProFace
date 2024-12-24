from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd
import random

def init_ff(phase,level='frame',n_frames=8):
    dataset_path='data/FaceForensics++/original_sequences/youtube/c23/frames/'
    
    image_list=[]
    label_list=[]

    folder_list = sorted(glob(dataset_path+'*'))
    filelist = []
    list_dict = json.load(open(f'./FF++/split/{phase}.json','r'))
    for i in list_dict:
        filelist+=i
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

    if level =='video':
        label_list = [0]*len(folder_list)
        return folder_list,label_list

    for i in range(len(folder_list)):

        images_temp=sorted(glob(folder_list[i]+'/*.png'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
        image_list+=images_temp
        label_list+=[0]*len(images_temp)

    return image_list, label_list


def init_ff_all(phase,level='frame',n_frames=8):
    dataset_path='data/FaceForensics++/original_sequences/youtube/c23/frames/'
    fake_dataset_path = 'data/FaceForensics++/manipulated_sequences/'+ 'Deepfakes/' + 'c23/' + 'frames/' 
    fake_dataset_path2 = 'data/FaceForensics++/manipulated_sequences/'+ 'Face2Face/' + 'c23/' + 'frames/'
    fake_dataset_path3 = 'data/FaceForensics++/manipulated_sequences/'+ 'FaceSwap/' + 'c23/' + 'frames/'
    fake_dataset_path4 = 'data/FaceForensics++/manipulated_sequences/'+ 'NeuralTextures/' + 'c23/' + 'frames/'
    image_list=[]
    label_list=[]
    
    fake_image_list=[]
    fake_label_list=[]
    
    folder_list = sorted(glob(dataset_path+'*'))
    folder_list_fake = sorted(glob(fake_dataset_path+'*'))
    folder_list_fake2 = sorted(glob(fake_dataset_path2+'*'))
    folder_list_fake3 = sorted(glob(fake_dataset_path3+'*'))
    folder_list_fake4 = sorted(glob(fake_dataset_path4+'*'))
    filelist = []
    filelist_fake = []
    list_dict = json.load(open(f'./FF++/split/{phase}.json','r'))
    for i in list_dict:
        filelist+=i
        filelist_fake.append(i[0]+'_'+i[1])
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
    folder_list_fake = [i for i in folder_list_fake if os.path.basename(i) in filelist_fake]
    folder_list_fake2 = [i for i in folder_list_fake2 if os.path.basename(i) in filelist_fake]
    folder_list_fake3 = [i for i in folder_list_fake3 if os.path.basename(i) in filelist_fake]
    folder_list_fake4 = [i for i in folder_list_fake4 if os.path.basename(i) in filelist_fake]
    if level =='video':
        label_list=[0]*len(folder_list)
        return folder_list, label_list

    for i in range(len(folder_list)):
        images_temp=sorted(glob(folder_list[i]+'/*.png'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
        image_list+=images_temp
        label_list+=[0]*len(images_temp)

    for i in range(len(folder_list_fake)):
        images_temp=sorted(glob(folder_list_fake[i]+'/*.png'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
        fake_image_list+=images_temp
        fake_label_list+=[1]*len(images_temp)

    for i in range(len(folder_list_fake2)):
        images_temp=sorted(glob(folder_list_fake2[i]+'/*.png'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
        fake_image_list+=images_temp
        fake_label_list+=[1]*len(images_temp)
        
    for i in range(len(folder_list_fake3)):
        images_temp=sorted(glob(folder_list_fake3[i]+'/*.png'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
        fake_image_list+=images_temp
        fake_label_list+=[1]*len(images_temp)
    
    for i in range(len(folder_list_fake4)):
        images_temp=sorted(glob(folder_list_fake4[i]+'/*.png'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
        fake_image_list+=images_temp
        fake_label_list+=[1]*len(images_temp)
        
    return image_list, label_list, fake_image_list, fake_label_list