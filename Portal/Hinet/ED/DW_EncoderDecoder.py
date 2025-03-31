import cv2
import torchvision

from . import *
from .Encoder_U import DW_Encoder
from .Decoder_U import DW_Decoder

# from .Noise import Noise
from .Random_Noise import Random_Noise
# from .noise_layers.Talk_to_Edit_main.TTS import TalkSwap
# from simswap.obfuscate import SimSwap
# from .noise_layers.FaceShifter.face_shifter import FaceShifter
from .noise_layers.face_identity_transformer_master.FIT import FITSwap


class DW_EncoderDecoder(nn.Module):
    
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder):
        super(DW_EncoderDecoder, self).__init__()

        self.encoder = DW_Encoder(message_length, attention=attention_encoder)
        self.noise = Random_Noise(noise_layers_R + noise_layers_F, len(noise_layers_R), len(noise_layers_F))

        self.decoder_C = DW_Decoder(message_length, attention=attention_decoder)

        # self.ts_swapmodel = TalkSwap()  # 输入batch必须是1
        # self.simswap = SimSwap()  # 输入batch必须是1
        # self.faceshifter_model = FaceShifter()
        self.fitswap_mode = FITSwap()

    def image_blur(self, steg_img, blur_sigma):
        blur_transform = kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(blur_sigma, blur_sigma), p=1)
        return blur_transform(steg_img)

    def image_gaussnoise(self, image, std=0.1):
        transform_gauss = kornia.augmentation.RandomGaussianNoise(mean=0, std=std, p=1)
        return transform_gauss(image)

    def image_identity(self, image, param=None):
        return image

    def image_resize(self, steg_img, down_scale=0.5):
        image = steg_img
        down = F.interpolate(image, size=(int(down_scale * image.shape[2]), int(down_scale * image.shape[3])),
                             mode='nearest')
        up = F.interpolate(down, size=(image.shape[2], image.shape[3]), mode='nearest')
        return up

    # cv2 真实jpeg压缩
    def image_jpeg(self, img, quality):
        B, _, _, _ = img.shape
        img = img.mul(255).add_(0.5).clamp_(0, 255)
        quality = torch.tensor(quality).repeat(B, 1)

        # Init list to store compressed images
        image_rgb_jpeg = []
        # Code each image
        for index in range(B):
            # Make encoding parameter
            encode_parameters = (int(cv2.IMWRITE_JPEG_QUALITY), int(quality[index].item()))
            # Encode image note CV2 is using [B, G, R]
            _, encoding = cv2.imencode(".jpeg", img[index].flip(0).permute(1, 2, 0).cpu().numpy(), encode_parameters)
            image_rgb_jpeg.append(torch.from_numpy(cv2.imdecode(encoding, 1)).permute(2, 0, 1).flip(0))

        # Stack images
        image_rgb_jpeg = torch.stack(image_rgb_jpeg, dim=0).float() / 255

        return image_rgb_jpeg

    def rand_distortion(self, img, step):
        # 定义所有变换方法和参数 ,共27中变换方法
        transformations = [
            (self.image_blur, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]),
            (self.image_gaussnoise, [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]),
            (self.image_resize, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
            (self.image_jpeg, [50, 75, 95]),
            (self.image_identity, [None])
        ]

        # 生成一个扁平化的变换方法和参数列表
        flat_transformations = [(func, param) for func, params in transformations for param in params]

        # 根据图片的下标选择变换方法和参数
        total_transforms = len(flat_transformations)
        index = step % total_transforms
        func, param = flat_transformations[index]

        # 执行选择的变换
        transformed_img = func(img, param)

        return transformed_img

    def denormalize(self, x):
        return (x + 1) / 2

    def forward(self, image, message, step):
        encoded_image = self.encoder(image, message)

        # torchvision.utils.save_image((encoded_image + 1) / 2,"/home/ysc/SepMark/network/noise_layers/Talk_to_Edit_main/results/encoded_image.png")
        # noised_image_C, noised_image_R, noised_image_F = self.noise([encoded_image, image, mask])

        # noised_image_R = self.rand_distortion(encoded_image.clone().detach(), step=step).to(image.device)

        # noised_image_R = encoded_image

        noised_image_R = self.image_jpeg(encoded_image, quality=75).to(image.device)
        # noised_image_R = self.image_gaussnoise(encoded_image, std=0.03).to(image.device)
        # noised_image_R = self.image_resize(encoded_image, down_scale=0.7).to(image.device)
        # noised_image_R = self.image_blur(encoded_image, blur_sigma=1.5).to(image.device)
        #
        noised_image_F = self.fitswap_mode(encoded_image.clone().detach())

        noised_image_F = self.image_jpeg(noised_image_F.clone().detach(), quality=75).to(image.device)

        # noised_image_F = self.image_gaussnoise(noised_image_F.clone().detach(), std=0.03).to(image.device)
        # noised_image_F = self.image_resize(noised_image_F.clone().detach(), down_scale=0.7).to(image.device)
        # noised_image_F = self.image_blur(noised_image_F.clone().detach(), blur_sigma=1.5).to(image.device)

        # torchvision.utils.save_image((noised_image_F + 1) / 2,"/home/ysc/SepMark/network/noise_layers/Talk_to_Edit_main/results/noised_image_F.png")

        # noised_image_F = self.rand_distortion(noised_image_F.clone().detach(), step=step).to(image.device)

        forward_image = image.clone().detach()

        noised_image_gap_R = noised_image_R.clamp(-1, 1) - forward_image
        noised_image_gap_F = noised_image_F.clamp(-1, 1) - forward_image

        noised_image_R = image + noised_image_gap_R
        noised_image_F = image + noised_image_gap_F

        decoded_message_R = self.decoder_RF(noised_image_R)
        decoded_message_F = self.decoder_RF(noised_image_F)

        return encoded_image, decoded_message_R, decoded_message_F
