import torch
import random
import argparse
from tqdm import tqdm
from datasets import *
from preprocess import extract_frames
sys.path.append('/media/Data8T/hanss/code/SelfBlendedBaseline/SBI-diffImages')
from src.retinaface.pre_trained_models import get_model
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def save_face(args):

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff()
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

    i=0
    for filename in tqdm(video_list):
        i+=1
        output_list=[]
        idx_lists = []

        try:
            face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)
            output_list = output_list + face_list
            idx_lists = idx_lists + idx_list
        except:
            print('123')
        output = np.array(output_list)
        idx = np.array(idx_lists)
        face_path = args.save_path+str(i)+'.npy'
        idx_path = args.save_path+str(i)+'_idx.npy'
        np.save(face_path, output)
        np.save(idx_path, idx)
    target_path = args.save_path + 'target_list.npy'
    np.save(target_path, np.array(target_list))
    print('finish')


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
    parser.add_argument('-s',dest='save_path',default='/media/Data8T/hanss/dataset/FF++/test_data/',type=str)
    parser.add_argument('-d',dest='dataset',default='FF',type=str)
    parser.add_argument('-s',dest='save_path',default='/media/Data8T/hanss/dataset/FFIW/test_data/',type=str)
    parser.add_argument('-d',dest='dataset',default='FFIW',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    save_face(args)  #保存提取的人脸

