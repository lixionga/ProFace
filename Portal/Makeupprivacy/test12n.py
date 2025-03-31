import os
import time
from PIL import Image
import sys




import torch.nn.functional as F
import torch
import torchvision.transforms as transforms

from torchvision.utils import save_image
from tqdm import tqdm

from util.util import *
from options.base_options import BaseOptions
from models.pix2pix_model import Pix2PixModel

import Pretrained_FR_Models.irse as irse
import Pretrained_FR_Models.facenet as facenet
import Pretrained_FR_Models.ir152 as ir152

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.networks.face_parsing.parsing_model import BiSeNet

import random

#log record


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

class Test_top_Options(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--test_name', type=str, default='1_facenet_multiscale=2', help='Overridden description for test',dest='name')
        parser.add_argument("--lfw_data_path", default="./Dataset-test/lfw112x112_500",help="path to lfw data images")
        parser.add_argument("--celeba_data_path", default="./Dataset-test/celeba_500_crop",help="path to lfw data images")       
        parser.add_argument("--reference_dir", default="./Dataset-test/reference",help="path to reference images")
        parser.add_argument('--which_epoch', type=str, default='latest',help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--beyond_mt', default='True',help='Want to transfer images that are not included in MT dataset, make sure this is Ture')
        parser.add_argument('--demo_mode', type=str, default='normal',help='normal|interpolate|removal|multiple_refs|partly')
        parser.add_argument('--model_name_list', type=list, default=['irse50', 'facenet', 'mobile_face','ir152'],help='fr model list')
        parser.add_argument('--model_name', type=str, default='irse50' ,help='fr model list')
        self.isTrain = False
        return parser
opt = Test_top_Options().parse()



test_models = {}

for model_name in opt.model_name_list:
    if model_name == 'ir152':
        test_models[model_name] = []
        test_models[model_name].append((112, 112))
        fr_model = ir152.IR_152((112, 112))
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
        fr_model.to(device)
        fr_model.eval()
        test_models[model_name].append(fr_model)
    if model_name == 'irse50':
        test_models[model_name] = []
        test_models[model_name].append((112, 112))
        fr_model = irse.Backbone(50, 0.6, 'ir_se')
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/irse50.pth'))
        fr_model.to(device)
        fr_model.eval()
        test_models[model_name].append(fr_model)
    if model_name == 'facenet':
        test_models[model_name] = []
        test_models[model_name].append((160, 160))
        fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/facenet.pth'))
        fr_model.to(device)
        fr_model.eval()
        test_models[model_name].append(fr_model)
    if model_name == 'mobile_face':
        test_models[model_name] = []
        test_models[model_name].append((112, 112))
        fr_model = irse.MobileFaceNet(512)
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/mobile_face.pth'))
        fr_model.to(device)
        fr_model.eval()
        test_models[model_name].append(fr_model)

def recognize_face(model_name,image):
    size = test_models[model_name][0]
    model = test_models[model_name][1]
    embbeding = model((F.interpolate(image, size=size, mode='bilinear')))       
    return embbeding

def get_similarity_probe_gallery(probe_image,gallery_paths):  
    probe_embedding = recognize_face(opt.model_name, probe_image)
    similarities=[]
    gallery_paths_list=[]
    for path in os.listdir(gallery_paths):
        gallery_paths_list.append(os.path.join(gallery_paths, path))
        img = read_img(os.path.join(gallery_paths, path), 0.5, 0.5, device)
        gallery_embedding=recognize_face(opt.model_name,img)
        similaritiy=cos_simi(probe_embedding,gallery_embedding)
        similaritiy=similaritiy.cpu().detach().numpy()
        similarities.append(similaritiy)
    return similarities

def get_similarity_probe_gallery(probe_image,other_image):  
    probe_embedding = recognize_face(opt.model_name,probe_image)
    other_embedding = recognize_face(opt.model_name,other_image)
    probe_other_similarity=cos_simi(probe_embedding,other_embedding).cpu().detach().numpy()
    return probe_other_similarity


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])

trans = transforms.Compose([transforms.ToTensor(),normalize])

def denorm(tensor):
    device = tensor.device
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

model = Pix2PixModel(opt)

if len(opt.gpu_ids) > 0:
            model = DataParallelWithCallback(model,device_ids=opt.gpu_ids)
model.eval()

n_classes = 19
parsing_net = BiSeNet(n_classes=n_classes)
parsing_net.load_state_dict(torch.load('./models/networks/face_parsing/79999_iter.pth'))
parsing_net.eval()
for param in parsing_net.parameters():
    param.requires_grad = False

def get_makeup_image(source_path,reference_path):
   
    source_name = source_path.replace('.jpg', '.png')
    source_path = source_path
    c= Image.open(source_path).convert("RGB")
    
    reference_name = reference_path.split('.')[0]
    reference_path = reference_path
    s = Image.open(reference_path).convert("RGB")

    
    height, width = c.size[0], c.size[1]
    c_m = c.resize((512, 512))
    s_m = s.resize((512, 512))
    c = c.resize((256, 256))
    s = s.resize((256, 256))
    # print(c.size)
    c_tensor = trans(c).unsqueeze(0)
    s_tensor = trans(s).unsqueeze(0)
    c_m_tensor = trans(c_m).unsqueeze(0)
    s_m_tensor = trans(s_m).unsqueeze(0)

    x_label = parsing_net(c_m_tensor)[0]
    y_label = parsing_net(s_m_tensor)[0]
    x_label = F.interpolate(x_label, (256, 256), mode='bilinear', align_corners=True)
    y_label = F.interpolate(y_label, (256, 256), mode='bilinear', align_corners=True)
    x_label = torch.softmax(x_label, 1)
    y_label = torch.softmax(y_label, 1)

    nonmakeup_unchanged = (x_label[0, 0, :, :] + x_label[0, 4, :, :] + x_label[0, 5, :, :] + x_label[0, 11, :,:] + x_label[0, 16,:,:] + x_label[0, 17, :,:]).unsqueeze(0).unsqueeze(0)
    makeup_unchanged = (y_label[0, 0, :, :] + y_label[0, 4, :, :] + y_label[0, 5, :, :] + y_label[0, 11, :,:] + y_label[0, 16, :,:] + y_label[0,17, :,:]).unsqueeze(0).unsqueeze(0)

    input_dict = {'nonmakeup': c_tensor,
                    'makeup': s_tensor,
                    'label_A': x_label,
                    'label_B': y_label,
                    'makeup_unchanged': makeup_unchanged,
                    'nonmakeup_unchanged': nonmakeup_unchanged
                    }

    synthetic_image = model([input_dict], mode='inference')
    makeup_image=synthetic_image[0]
    makeup_image = F.interpolate(makeup_image, (256, 256 * height // width), mode='bilinear', align_corners=False)
    return makeup_image
   


def test(test_data_path):
    
    path_list=os.listdir(test_data_path)
    path_list.sort()  
    probe_image_path_list=[]
    gallery_image_path_list=[]
    for file in path_list:
        file_path = os.path.join(test_data_path, file)
        file_list=os.listdir(file_path)
        file_list.sort(key= lambda x:int(x[-5])) 
        
        img_1_path=os.path.join(file_path,file_list[0])
        img_2_path=os.path.join(file_path,file_list[1])
        
        probe_image_path_list.append(img_1_path)
        gallery_image_path_list.append(img_2_path)
    u_top1=0
    u_top5=0
    reference_dir_list = os.listdir(opt.reference_dir)
    if(test_data_path==opt.lfw_data_path):
        slicce_int=-5
    elif(test_data_path==opt.celeba_data_path):
        slicce_int=-10
    for probe_img_path in tqdm(probe_image_path_list):
        probe_gallery_cos_list=[]
        reference_list_choice=random.choice(reference_dir_list)
        reference_path=os.path.join(opt.reference_dir, reference_list_choice)
        probe_image = get_makeup_image(probe_img_path, reference_path)
        for gallery_img_path in tqdm(gallery_image_path_list):
            if probe_img_path[:slicce_int]==gallery_img_path[:slicce_int]:              
                other_image = read_img(gallery_img_path, 0.5, 0.5, device)
                probe_other_cos=get_similarity_probe_gallery(probe_image,other_image)
            else:
                gallery_image = read_img(gallery_img_path, 0.5, 0.5, device)
                probe_gallery_cos_list.append(get_similarity_probe_gallery(probe_image,gallery_image))
              
        probe_gallery_cos_list.sort(reverse=True)

        if probe_other_cos<=probe_gallery_cos_list[0]:
            u_top1+=1
        if probe_other_cos<=probe_gallery_cos_list[4]:
            u_top5+=1

    print("PSR_u_top1 :{:.5f}".format(u_top1/len(probe_image_path_list)) )
    print("PSR_u_top5 :{:.5f}".format(u_top5/len(probe_image_path_list)) )
    

if __name__ == '__main__':
    type = sys.getfilesystemencoding()
    sys.stdout = Logger("./log")
    test(opt.celeba_data_path)


















































