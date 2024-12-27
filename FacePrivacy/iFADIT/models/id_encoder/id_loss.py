import torch
from torch import nn
from PIL import Image
import sys

sys.path.append('..')
from models.UtilModels.encoders.model_irse import Backbone
import torch.nn.functional as F

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class IDLoss(nn.Module):
    def __init__(self, pretrained_model_path):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(pretrained_model_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target

        return loss / n_samples



class IDLossExtractor(torch.nn.Module):
    def __init__(self, pretrained_model_path, margin, requires_grad=False):
        super(IDLossExtractor, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(pretrained_model_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.margin=margin
        self.margin=self.margin if self.margin is not None  else 0.1 

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats
    
    def calculate_loss(self,features1,features2):

        assert len(features1)==len(features2)
        
        percept_loss,id_loss=0,0
        for i in range(len(features1)-1):
            #print(features1[i].shape,features2[i].shape)
            percept_loss+=F.l1_loss(features1[i],features2[i])
        

        rev_id_loss=torch.cosine_similarity(features1,features2).clamp(min=self.margin).mean()
        id_loss=(1-torch.cosine_similarity(features1,features2).clamp(min=self.margin)).mean()
        # print(f'festure1 is {features1.shape}')
        inter_loss = torch.tril(features1@(features1.permute(1,0)), diagonal=-1)
        # print(inter_loss)
        inter_loss = torch.where(inter_loss<0.1, inter_loss-inter_loss, inter_loss)
        # print(inter_loss)
        inter_loss = torch.sum(inter_loss)

        return id_loss, rev_id_loss, percept_loss, inter_loss


    def forward(self,img1,img2):

        features1=self.extract_feats(img1)
        features2=self.extract_feats(img2)
        
        return self.calculate_loss(features1,features2)
    

class inter_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        
        loss = torch.tril(input_tensor@input_tensor.permute(1,0), diagonal=-1)
        print(loss)
        loss = torch.where(loss<0.1, loss-loss, loss)
        print(loss)
        loss = torch.sum(loss)

        return loss



if __name__ == "__main__":

    import torch
    test_tensor = torch.randn((4,4))
    print(test_tensor)
    loss = inter_loss()
    out = loss(test_tensor)
    print(out)
    a

    import torchvision.transforms as transforms
    import cv2 as cv

    dive_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),

                                         ]
                                        )

    img1 = cv.imread('/userHOME/yl/lk_data/CASIA-WebFace/0000107/007.jpg')
    img2 = cv.imread('/userHOME/yl/lk_data/CASIA-WebFace/0000107/012.jpg')
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    # print(img.shape)  # numpy数组格式为（H,W,C）
    transf = transforms.ToTensor()

    img1_tensor = transf(img1)  # tensor数据格式是torch(C,H,W)
    img2_tensor = transf(img2)
    # print(img1_tensor.size())
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    img1_tensor = transform(img1_tensor)
    img2_tensor = transform(img2_tensor)
    id_fir = img1_tensor
    id_sec = img2_tensor

    # id_fir = np.random.rand(4, 3, 256, 256)
    # id_fir = torch.from_numpy(id_fir)
    id_fir = id_fir.to(torch.float32)
    id_fir = id_fir.to('cuda:0')
    id_fir = id_fir.unsqueeze(0)
    print(f'id_fir shape {id_fir.shape}')

    id_sec = id_sec.to(torch.float32)
    id_sec = id_sec.to('cuda:0')
    id_sec = id_sec.unsqueeze(0)
    print(f'id_sec shape {id_sec.shape}')


##########add detection##############
    encoder = IDLoss(pretrained_model_path='/home/yl/lk/code/ID-dise4/model_ir_se50.pth')
    encoder = encoder.to('cuda:0')
    encoder = encoder.eval()
    
    with torch.no_grad():
        id_fir_feature = encoder.extract_feats(id_fir)
        



#####################################
    # print(id_fir)
    #
    #
    id_crop = dive_transform(id_fir)
    id_sec_crop = dive_transform(id_sec)
    # print(f'id_fir {id_fir.shape}, {type(id_fir)}')
    encoder = IDLoss(pretrained_model_path='/home/yl/lk/code/ID-dise4/model_ir_se50.pth')
    encoder = encoder.to('cuda:0')
    encoder = encoder.eval()
    print(id_crop)
    with torch.no_grad():
        id_fir_feature = encoder.extract_feats(id_fir)
        id_fir_feature_crop = encoder.extract_feats(id_crop)
        id_sec_feature = encoder.extract_feats(id_sec)
        id_sec_feature_crop = encoder.extract_feats(id_sec_crop)

    #     loss1 = encoder(id_fir, id_crop)
    #
    # print(f'loss1  is{loss1}')

    cosine_similarity = torch.cosine_similarity(id_fir_feature, id_sec_feature, dim=0).mean()
    print(f'cosine_similarity  is{cosine_similarity}')

    calc_loss = IDLossExtractor('/home/yl/lk/code/ID-dise4/model_ir_se50.pth', margin=0.1)
    calc_loss = calc_loss.to('cuda:0')
    
    temp_id_loss,_,_ = calc_loss(id_fir, id_sec)
    print(f'temp_id_loss is  {temp_id_loss} \n, shape is {temp_id_loss.shape}')

    
    