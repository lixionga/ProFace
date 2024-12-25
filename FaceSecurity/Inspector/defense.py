import torch
import torch.nn as nn
from model.network.xception import Generator
from model.network.models import model_selection
import random

class Inspector(nn.Module):
    def __init__(self, pth_rec, pth_cor, pth_threshold):
        super(Inspector, self).__init__()
        self.reconstructer = Generator().eval()
        self.corrector, *_ = model_selection(modelname='resnet18', num_out_classes=2)
        self.autothresholder, *_ = model_selection(modelname='resnet18', num_out_classes=2)
        
        self.reconstructer.load_state_dict(torch.load(pth_rec), strict=False)
        self.corrector.load_state_dict(torch.load(pth_cor), strict=False)
        self.autothresholder.load_state_dict(torch.load(pth_threshold), strict=False)
        
        self.corrector.eval()
        self.autothresholder.eval()
        
    def forward(self, x, feat, pred_cls):
        rec = self.reconstructer(feat, x)
        pred = self.autothresholder(rec)[0]
        pred_label = True
        
        if torch.argmax(pred, dim=-1):
            pred_label = False
                   
        else:
            pred_cor = torch.sigmoid(self.corrector(x)[0]).unsqueeze(1)
            pred_cor = pred_cor @ torch.softmax(pred_cls, dim=-1).unsqueeze(-1)
                        
            if pred_cor.squeeze(-1).squeeze(-1) > 0.5:
                pred_cls = pred_cls - torch.sum(pred_cls)/2
                pred_cls = torch.sum(pred_cls) - pred_cls
                pred_cls = 1/pred_cls
                
            else:
                if random.random() > 0.5:
                    pred_cls = pred_cls * random.random()
                else:
                    pred_cls = pred_cls * random.randint(2,100)
            
        return pred_cls, pred_label


class EntireModel(nn.Module):
    def __init__(self, forgery_detector, inspector):
        super(EntireModel, self).__init__()
        self.model = forgery_detector.eval()
        self.inspector = inspector
        
    def forward(self, img):
        pred_adv, rec = self.forgery_detector(img)
        pred_defense, pred_model = self.inspector(img, rec, pred_adv)
        
        return pred_defense, pred_model