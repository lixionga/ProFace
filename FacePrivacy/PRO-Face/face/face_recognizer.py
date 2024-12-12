'''
Get the face recognizer model and the image size that fits the recognizer
'''
import os
import torch
from .inception_resnet_v1 import InceptionResnetV1
from .iresnet import iresnet100
from .MobileFaceNet import MobileFacenet
from .CBAM import CBAMResNet
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
import pickle


dir_checkpoints = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints')


# Face models: ['MobileFaceNet', 'InceptionResNet', 'IResNet50', 'IResNet100', 'SEResNet50']
model_options = {
    'InceptionResNet': (InceptionResnetV1(classify=False, pretrained='vggface2'), 'vggface2.pt', 160),
    'IResNet100': (iresnet100(), 'iresnet100.pt', 112),
    'MobileFaceNet': (MobileFacenet(), 'CASIA_WebFace_MobileFaceNet/Iter_64000_net.pth', 112),
    'IResNet50': (CBAMResNet(50, mode='ir'), 'CASIA_WebFace_ResNet50_IR/Iter_64000_net.pth', 112),
    'SEResNet50': (CBAMResNet(50, mode='ir_se'), 'CASIA_WebFace_SEResNet50_IR/Iter_64000_net.pth', 112),
}


def load_state_dict_pkl(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def rgb2bgr(x):
    return x[:, [2, 1, 0], :, :]


def get_recognizer(name='InceptionResNetV1'):
    grayscale = True if name == 'resnet18' else False

    recognizer, filename, img_size = model_options[name]
    model_path = os.path.join(dir_checkpoints, filename)
    recognizer.load_state_dict(torch.load(model_path))

    input_transforms = [
        T.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5)
    ]

    # resize = transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC)
    resize_transforms = [
        T.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
    ]

    recognizer.name = name
    recognizer.img_size = img_size
    recognizer.grayscale = grayscale
    recognizer.trans = T.Compose(input_transforms)
    recognizer.resize = T.Compose(resize_transforms)

    return recognizer


num_params = lambda model : sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    for model_name in list(model_options.keys()):
        model = get_recognizer(model_name)
        print('{:<20}: {}'.format(model_name, num_params(model)))
