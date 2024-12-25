import torch
import numpy as np
import random
from model import Detector
import argparse
from tqdm import tqdm
from preprocess import extract_face_test
from datasets import *
from sklearn.metrics import roc_auc_score
import warnings

from sklearn.metrics import average_precision_score as ap

warnings.filterwarnings('ignore')

def main(args):

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    
    model.load_state_dict(cnn_sd) 
    model.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        if args.subname is None:
            video_list,target_list=init_ff()
        else:
            video_list,target_list=init_ff(dataset = args.subname)
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    output_list=[]

    for filename in tqdm(video_list):
        
        try:
            if args.dataset == 'CDF':
                filename = filename.replace('videos/', '')
                name = filename[filename.find('Celeb'):].split('/') 
                face_pth = './FPG/data/' + name[0] + '/' + name[1] + '/retina/' + name[-1].replace('.mp4', '')

            elif args.dataset == 'FF':
                filename = filename.replace('videos/', '')
                name = filename[filename.find('FF++') + 4:].split('/') # 
                face_pth = './FPG/data/FaceForensics++/' + name[1] + '/' + name[2] + '/' + name[3] + '/retina/' + name[-1].replace('.mp4', '')
            
            elif args.dataset == 'DFD':
                filename = filename.replace('videos/', '')
                name = filename[filename.find('FF++') + 4:].split('/') # 
                face_pth = './FPG/data/FaceForensics++/' + name[1] + '/' + name[2] + '/' + name[3] + '/retina/' + name[-1].replace('.mp4', '') # raw

            elif args.dataset == 'FFIW':
                filename = filename.replace('videos/', '')
                name = filename[filename.find('FFIW'):].split('/') # 
                face_pth = './FPG/data/' + name[0] + '/' + name[1] + '/' + name[2] + '/' + name[3] + '/retina/' + name[-1].replace('.mp4', '')
                
            elif args.dataset == 'DFDC':
                filename = filename.replace('videos/', '')
                name = filename[filename.find('DFDC'):].split('/') # 
                face_pth = './FPG/data/' + name[0] + '/' + name[1] + '/retina/' + name[-1].replace('.mp4', '')

            elif args.dataset == 'DFDCP':
                filename = filename.replace('videos/', '')
                if 'original' not in filename:
                    name = filename[filename.find('DFDCP'):].split('/') # 
                    face_pth = './FPG/data/' + name[0] + '/' + name[1] + '/' + name[2] + '/' + name[3] + '/'+ name[-1].replace('.mp4', '') + '/retina/' + name[-1].replace('.mp4', '')
                else:
                    name = filename[filename.find('DFDCP'):].split('/') # 
                    face_pth = './FPG/data/' + name[0] + '/original_videos/'  + name[1].replace('original_','') + '/' + name[2].replace('.mp4', '') + '/retina/' + name[-1].replace('.mp4', '')
            
            
            face_list, idx_list = extract_face_test(face_pth)
                
            with torch.no_grad():
                img=torch.tensor(face_list).to(device).float()/255
                pred=model(img).softmax(1)[:,1]
            
            pred_list=[]
            idx_img=-1
            
            for i in range(len(pred)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res=np.zeros(len(pred_list))
            
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            pred=pred_res.mean()
            
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)

    
    auc = roc_auc_score(target_list, output_list) 
    ap_score  = ap(np.array(target_list), np.array(output_list))

        
    if args.subname is None:
        print(f'Weight:{args.weight_name} | {args.dataset}| AUC: {auc:.4f}')
        print(f'Weight:{args.weight_name} | {args.dataset}| AP: {ap_score:.4f}')
    else:
        print(f'Weight:{args.weight_name} | {args.dataset}| {args.subname}| AUC: {auc:.4f}')
        print(f'Weight:{args.weight_name} | {args.dataset}| {args.subname}| AP: {ap_score:.4f}')

    
if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-sub',dest='subname',default=None,type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)