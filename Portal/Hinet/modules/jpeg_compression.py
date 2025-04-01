import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as c
device = torch.device(c.device if torch.cuda.is_available() else "cpu")


def dct_coeff(n, k, N):
    # 该函数返回一个值，表示离散余弦变换（Discrete Cosine Transform，DCT）中的一个系数。
    return np.cos(np.pi / N * (n + 1. / 2.) * k)


def idct_coeff(n, k, N):
    # 该函数返回一个值，表示离散余弦逆变换（Inverse Discrete Cosine Transform，IDCT）中的一个系数。
    return (int(0 == n) * (- 1 / 2) + np.cos(
        np.pi / N * (k + 1. / 2.) * n)) * np.sqrt(1 / (2. * N))


def gen_filters(size_x: int, size_y: int, dct_or_idct_fun: callable) -> np.ndarray:
    tile_size_x = 8
    filters = np.zeros((size_x * size_y, size_x, size_y))
    for k_y in range(size_y):
        for k_x in range(size_x):
            for n_y in range(size_y):
                for n_x in range(size_x):
                    filters[k_y * tile_size_x + k_x, n_y, n_x] = dct_or_idct_fun(n_y, k_y, size_y) * dct_or_idct_fun(
                        n_x,
                        k_x,
                        size_x)
    return filters


def rgb2yuv(image_rgb, image_yuv_out):
    """ Transform the image from rgb to yuv """
    image_yuv_out[:, 0, :, :] = 0.299 * image_rgb[:, 0, :, :].clone() + 0.587 * image_rgb[:, 1, :,
                                                                                :].clone() + 0.114 * image_rgb[:, 2, :,
                                                                                                     :].clone()
    image_yuv_out[:, 1, :, :] = -0.14713 * image_rgb[:, 0, :, :].clone() + -0.28886 * image_rgb[:, 1, :,
                                                                                      :].clone() + 0.436 * image_rgb[:,
                                                                                                           2, :,
                                                                                                           :].clone()
    image_yuv_out[:, 2, :, :] = 0.615 * image_rgb[:, 0, :, :].clone() + -0.51499 * image_rgb[:, 1, :,
                                                                                   :].clone() + -0.10001 * image_rgb[:,
                                                                                                           2, :,
                                                                                                           :].clone()


def yuv2rgb(image_yuv, image_rgb_out):
    """ Transform the image from yuv to rgb """
    image_rgb_out[:, 0, :, :] = image_yuv[:, 0, :, :].clone() + 1.13983 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 1, :, :] = image_yuv[:, 0, :, :].clone() + -0.39465 * image_yuv[:, 1, :,
                                                                           :].clone() + -0.58060 * image_yuv[:, 2, :,
                                                                                                   :].clone()
    image_rgb_out[:, 2, :, :] = image_yuv[:, 0, :, :].clone() + 2.03211 * image_yuv[:, 1, :, :].clone()


def get_jpeg_yuv_filter_mask(image_shape: tuple, window_size: int, keep_count: int):
    mask = np.zeros((window_size, window_size), dtype=np.uint8)

    index_order = sorted(((x, y) for x in range(window_size) for y in range(window_size)),
                         key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

    for i, j in index_order[0:keep_count]:
        mask[i, j] = 1

    return np.tile(mask, (int(np.ceil(image_shape[0] / window_size)),
                          int(np.ceil(image_shape[1] / window_size))))[0: image_shape[0], 0: image_shape[1]]


# 模拟jpeg
# def image_jpeg(steg_img):
#     jpeg_res = None
#     for i in range(steg_img.shape[0]):
#         steg_img_i = steg_img[i, :, :, :]
#         steg_img_i = steg_img_i.unsqueeze(0)
#         jpeg_input = steg_img_i.clone()
#         jpeg_output = JpegCompression((25, 9, 9))(jpeg_input)
#         if jpeg_res == None:
#             jpeg_res = jpeg_output
#         else:
#             jpeg_res = torch.cat((jpeg_res, jpeg_output), 0)
#
#     return jpeg_res.to(device)

class JpegCompression(nn.Module):
    def __init__(self, yuv_keep_weights=(25, 9, 9)):
        super(JpegCompression, self).__init__()
        self.yuv_keep_weights = yuv_keep_weights
        self.dct_conv_weights = torch.tensor(gen_filters(8, 8, dct_coeff), dtype=torch.float32).to(device)
        self.dct_conv_weights.unsqueeze_(1)
        self.idct_conv_weights = torch.tensor(gen_filters(8, 8, idct_coeff), dtype=torch.float32).to(device)
        self.idct_conv_weights.unsqueeze_(1)
        self.jpeg_mask = None
        self.create_mask((1000, 1000))

    def create_mask(self, requested_shape):
        if self.jpeg_mask is None or requested_shape > self.jpeg_mask.shape[1:]:
            self.jpeg_mask = torch.empty((3,) + requested_shape, device=device)
            for channel, weights_to_keep in enumerate(self.yuv_keep_weights):
                mask = torch.from_numpy(get_jpeg_yuv_filter_mask(requested_shape, 8, weights_to_keep))
                self.jpeg_mask[channel] = mask

    def get_mask(self, image_shape):
        # if the mask is too small, create a new one
        if self.jpeg_mask.shape < image_shape:
            self.create_mask(image_shape)
        return self.jpeg_mask[:, :image_shape[1], :image_shape[2]].clone()

    def apply_conv(self, image, filter_type: str):

        if filter_type == 'dct':
            filters = self.dct_conv_weights
        elif filter_type == 'idct':
            filters = self.idct_conv_weights
        else:
            raise ('Unknown filter_type value.')

        image_conv_channels = []
        for channel in range(image.shape[1]):
            image_yuv_ch = image[:, channel, :, :].unsqueeze_(1)  # add channel dimension
            image_conv = F.conv2d(image_yuv_ch.clone(), filters, stride=8)
            image_conv = image_conv.permute(0, 2, 3, 1)
            image_conv = image_conv.view(image_conv.shape[0], image_conv.shape[1], image_conv.shape[2], 8, 8)
            image_conv = image_conv.permute(0, 1, 3, 2, 4)
            image_conv = image_conv.contiguous().view(image_conv.shape[0],
                                                      image_conv.shape[1] * image_conv.shape[2],
                                                      image_conv.shape[3] * image_conv.shape[4])

            image_conv.unsqueeze_(1)  # add channel dimension
            image_conv_channels.append(image_conv)
        image_conv_stacked = torch.cat(image_conv_channels, dim=1)  # stack channels
        return image_conv_stacked

    def forward(self, img):
        # 1. 对图片进行预处理（长、宽补充为8的倍数）
        noised_image = img.clone()
        # pad the image so that we can do dct on 8x8 blocks
        pad_height = (8 - noised_image.shape[2] % 8) % 8
        pad_width = (8 - noised_image.shape[3] % 8) % 8
        noised_image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(noised_image)

        # 2. 将图片转换为YUV颜色空间
        image_yuv = torch.empty_like(noised_image)
        rgb2yuv(noised_image, image_yuv)

        # 3. 对图片进行DCT变换
        image_dct = self.apply_conv(image_yuv, 'dct')

        # 4. 对图片进行量化
        mask = self.get_mask(image_dct.shape[1:])
        image_dct_mask = torch.mul(image_dct, mask)

        # 5. 对图片进行反DCT变换
        image_idct = self.apply_conv(image_dct_mask, 'idct')

        # 6. 将图片转换为RGB颜色空间
        image_ret_padded = torch.empty_like(image_dct)
        yuv2rgb(image_idct, image_ret_padded)

        # 7. 将图片去除预处理的补充部分
        img = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
              :image_ret_padded.shape[3] - pad_width].clone()

        return img
