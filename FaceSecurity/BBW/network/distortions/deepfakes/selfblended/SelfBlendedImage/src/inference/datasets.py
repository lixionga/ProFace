from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd


def init_ff(dataset='all',phase='test'):
	assert dataset in ['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	original_path='data/FaceForensics++/original_sequences/youtube/raw/videos/'
	folder_list = sorted(glob(original_path+'*'))

	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	filelist=[]
	for i in list_dict:
		filelist+=i
	image_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
	label_list=[0]*len(image_list)


	if dataset=='all':
		fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	else:
		fakes=[dataset]

	folder_list=[]
	for fake in fakes:
		fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/raw/videos/'
		folder_list_all=sorted(glob(fake_path+'*'))
		folder_list+=[i for i in folder_list_all if os.path.basename(i)[:3] in filelist]
	label_list+=[1]*len(folder_list)
	image_list+=folder_list
	return image_list,label_list



def init_dfd():
	real_path='data/FaceForensics++/original_sequences/actors/raw/videos/*.mp4'
	real_videos=sorted(glob(real_path))
	fake_path='data/FaceForensics++/manipulated_sequences/DeepFakeDetection/raw/videos/*.mp4'
	fake_videos=sorted(glob(fake_path))

	label_list=[0]*len(real_videos)+[1]*len(fake_videos)

	image_list=real_videos+fake_videos

	return image_list,label_list


def init_dfdc():
		
	label=pd.read_csv('data/DFDC/labels.csv',delimiter=',')
	folder_list=[f'data/DFDC/videos/{i}' for i in label['filename'].tolist()]
	label_list=label['label'].tolist()
	
	return folder_list,label_list


def init_dfdcp(phase='test'):

	phase_integrated={'train':'train','val':'train','test':'test'}

	all_img_list=[]
	all_label_list=[]

	with open('data/DFDCP/dataset.json') as f:
		df=json.load(f)
	fol_lab_list_all=[[f"data/DFDCP/{k.split('/')[0]}/videos/{k.split('/')[-1]}",df[k]['label']=='fake'] for k in df if df[k]['set']==phase_integrated[phase]]
	name2lab={os.path.basename(fol_lab_list_all[i][0]):fol_lab_list_all[i][1] for i in range(len(fol_lab_list_all))}
	fol_list_all=[f[0] for f in fol_lab_list_all]
	fol_list_all=[os.path.basename(p)for p in fol_list_all]
	folder_list=glob('data/DFDCP/method_*/videos/*/*/*.mp4')+glob('data/DFDCP/original_videos/videos/*/*.mp4')
	folder_list=[p for p in folder_list if os.path.basename(p) in fol_list_all]
	label_list=[name2lab[os.path.basename(p)] for p in folder_list]
	

	return folder_list,label_list




def init_ffiw():
	# assert dataset in ['real','fake']
	path='data/FFIW/FFIW10K-v1-release/'
	folder_list=sorted(glob(path+'source/val/videos/*.mp4'))+sorted(glob(path+'target/val/videos/*.mp4'))
	label_list=[0]*250+[1]*250
	return folder_list,label_list



def init_cdf():

	image_list=[]
	label_list=[]

	video_list_txt='data/Celeb-DF-v2/List_of_testing_videos.txt'
	with open(video_list_txt) as f:
		
		folder_list=[]
		for data in f:
			# print(data)
			line=data.split()
			# print(line)
			path=line[1].split('/')
			folder_list+=['data/Celeb-DF-v2/'+path[0]+'/videos/'+path[1]]
			label_list+=[1-int(line[0])]
		return folder_list,label_list
		


