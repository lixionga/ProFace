import random


def get_random_float(float_range: [float]):
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])


from .identity import Identity
from .crop import FaceCrop, FaceCropout, Dropout, FaceErase, FaceEraseout
from .salt_pepper import SaltPepper
from .jpeg import JpegTest
from .resize import Resize
from .kornia_noises import GaussianBlur, GaussianNoise, MedianBlur, Brightness, Contrast, Saturation, Hue, Rotation, Affine
############################################
from simswap.obfuscate import SimSwap
# from network.noise_layers.faceswap.face_swap import FaceSwap
# from .ganimation.main import GANimation
# from .stargan.main import StarGAN
#from .mobilefaceswap.image_test import MobileFaceSwap
#from .roop.ROOP import ROOP
