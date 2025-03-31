import torch
from torchvision import transforms

class BlurTransform(object):
    def __init__(self, kernel_size=31, sigma=4.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image):
        # 实现batch image_blur函数的图片模糊逻辑
        img_blurred_res = None
        for i in range(image.shape[0]):
            img_blurred_i = image[i, :, :, :]
            trans_blur = transforms.GaussianBlur(self.kernel_size, self.sigma)
            img_blurred_i = trans_blur(img_blurred_i).unsqueeze(0)
            if img_blurred_res == None:
                img_blurred_res = img_blurred_i
            else:
                img_blurred_res = torch.cat((img_blurred_res, img_blurred_i), 0)
        return img_blurred_res