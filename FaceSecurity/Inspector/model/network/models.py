import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network.xception import xception
import math
from model.vgg import vgg19
from model.efficientnet_pytorch import EfficientNet
from model.resnet import resnet18, resnet34, resnet50

def return_pytorch04_xception(pretrained=True):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            './model/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes))
                
        elif modelchoice == 'resnet50' or modelchoice == 'resnet34' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = resnet50(pretrained=True)
            if modelchoice == 'resnet34':
                self.model = resnet34(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'vgg19':
            if modelchoice == 'vgg19':
                self.model = vgg19(pretrained=True)
            if not dropout:
                self.model.classifier = self.classifier = nn.Sequential(
                        nn.Linear(512 * 7 * 7, 4096),
                        nn.ReLU(True),
                        nn.Linear(4096, 4096),
                        nn.ReLU(True),
                        nn.Linear(4096, num_out_classes))
            else:
                self.model.classifier =  nn.Sequential(
                        nn.Linear(512 * 7 * 7, 4096),
                        nn.ReLU(True),
                        nn.Dropout(p=dropout),
                        nn.Linear(4096, 4096),
                        nn.ReLU(True),
                        nn.Dropout(p=dropout),
                        nn.Linear(4096, num_out_classes))
        elif modelchoice == 'efficientnet-b4':
            if modelchoice == 'efficientnet-b4':
                self.model = EfficientNet.from_pretrained('efficientnet-b4')
            # Replace fc
            num_ftrs = self.model._fc.in_features
            if not dropout:
                self.model._fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model._fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None, transition=False):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        if not transition:
            return TransferModel(modelchoice='xception',
                                num_out_classes=num_out_classes), 299, \
                True, ['image'], None
        else:
            return TransferModel(modelchoice='xception',
                                num_out_classes=num_out_classes**2), 299, \
                True, ['image'], None
                
    elif modelname == 'resnet18':
        if not transition:
            return TransferModel(modelchoice='resnet18', dropout=dropout,
                                num_out_classes=num_out_classes), \
                224, True, ['image'], None
        else:
            return TransferModel(modelchoice='resnet18', dropout=dropout,
                                num_out_classes=num_out_classes**2), \
                224, True, ['image'], None
    elif modelname == 'resnet34':
        if not transition:
            return TransferModel(modelchoice='resnet34', dropout=dropout,
                                num_out_classes=num_out_classes), \
                224, True, ['image'], None
        else:
            return TransferModel(modelchoice='resnet34', dropout=dropout,
                                num_out_classes=num_out_classes**2), \
                224, True, ['image'], None
    elif modelname == 'resnet50':
        return TransferModel(modelchoice='resnet50', dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    elif modelname == 'vgg19':
        return TransferModel(modelchoice='vgg19', dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    elif modelname == 'efficientnet-b4':
        return TransferModel(modelchoice='efficientnet-b4', dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    else:
        raise NotImplementedError(modelname)