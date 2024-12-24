import torch
from torch import nn
import sys
from efficientnet_pytorch import EfficientNet
from face_net import iresnet18
from bnneck import BNClassifier
from sam import SAM
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class FeatureReconstructionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1):
        super(FeatureReconstructionNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim1)
        )

    def forward(self, x):

        return self.fc(x)


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        weight_path = 'weights/adv-efficientnet-b4-44fb3a87.pth'
        self.net=EfficientNet.from_pretrained("efficientnet-b4",weights_path=weight_path,advprop=True,num_classes=2)

        self.face_model = iresnet18()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weight = torch.load('weights/backbone.pth', map_location=device)
        # weight = torch.load('weights/backbone.pth')
        self.face_model.load_state_dict(weight)
        self.face_model = self.face_model.eval()


        self.bn = nn.BatchNorm1d(1792)
        self.bn.apply(weights_init_kaiming)
        self.cel = nn.CrossEntropyLoss()
        self.reshape_layer = nn.Linear(1792, 2304)
        self.fc_cat = BNClassifier(1792, 2)
        self.fc_cat1 = BNClassifier(1792, 2)
        self.mse = nn.MSELoss()
        self.fc = nn.Linear(512, 2)
        self.reconstruction_network = FeatureReconstructionNetwork(2304, 1024, 1792, )
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=0.001, momentum=0.9)

    def forward(self,x,mode='test'):
        if mode=='test':
            arti_feats = self.net.extract_features(x)
            arti_feats = self.net._avg_pooling(arti_feats)
            if self.net._global_params.include_top:
                arti_feats = arti_feats.flatten(start_dim=1)
                arti_feats = self.net._dropout(arti_feats)
                arti_feat_bn = self.bn(arti_feats)
                arti_feats_cls = self.net._fc(arti_feat_bn)

        return arti_feats_cls

