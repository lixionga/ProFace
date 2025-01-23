import os
import torch
import numpy as np
import random
from model import Detector
import argparse
from tqdm import tqdm
import logging
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

log_filename = '/media/Data8T/hanss/code/SelfBlendedBaseline/EG-deepfake-detection/output/test/inference.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_test(args, device, weight_file):
    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load(weight_file)["model"]
    model.load_state_dict(cnn_sd)
    auc = test(model, args, device)
    return auc


def test(model,args,device):

    model.eval()

    target_list = np.load(args.test_dataset_path+'target_list.npy')
    target_list = list(target_list)
    output_list=[]
    for i in tqdm(range(len(target_list))):
        i+=1
        try:
            face_path = args.test_dataset_path+str(i)+'.npy'
            idx_path = args.test_dataset_path+str(i)+'_idx.npy'
            face_list,idx_list = np.load(face_path),np.load(idx_path)

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

    auc=roc_auc_score(target_list,output_list)
    logging.info(f'{args.dataset}| AUC: {auc:.4f}')
    print(f'{args.dataset}| AUC: {auc:.4f}')
    return auc


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
    parser.add_argument('-w',dest='weight_name',default='/media/Data8T/hanss/code/SelfBlendedBaseline/FFIW-PFlevel-fusion/output/CDF-weight/sbi_base_10_27_00_08_35-0.4*distill+0.01*rc/weights/73_0.9528_val.tar',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    dataset_paths = [
        {'path': '/media/Data8T/hanss/dataset/FF++/test_data/', 'name': 'FF'},
        {'path': '/media/Data8T/hanss/DeepfakeBench-main/datasets/Celeb-DF-v2/test_data/', 'name': 'CDF'},
        {'path': '/media/Data8T/hanss/dataset/DFD-raw/test_data/', 'name': 'DFD'},
        {'path': '/media/Data8T/hanss/DFDC-test/test_data/', 'name': 'DFDC'},
        {'path': '/media/Data8T/hanss/dataset/DFDCP/test_data/', 'name': 'DFDCP'},
        {'path': '/media/Data8T/hanss/dataset/FFIW/test_data/', 'name': 'FFIW'},
    ]


    for dataset in dataset_paths:
        args.test_dataset_path = dataset['path']
        args.dataset = dataset['name']
        auc_list = []

        print(f'Testing with weight file: {args.weight_name}')
        logging.info(f'Testing with weight file: {args.weight_name}')
        auc = load_test(args, device, args.weight_name)



