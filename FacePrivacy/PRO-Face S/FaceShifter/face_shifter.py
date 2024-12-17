import torchvision.transforms as transforms
from FaceShifter.face_modules.model import Backbone
from FaceShifter.network.AEI_Net import *
from FaceShifter.face_modules.mtcnn import *
import cv2
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import config.config as c


detector = MTCNN()
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = c.DEVICE
G = AEI_Net(c_id=512)
G.eval()
G.load_state_dict(torch.load('FaceShifter/saved_models/G_latest.pth', map_location=device))
G = G.to(device)
arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('FaceShifter/face_modules/model_ir_se50.pth', map_location=device), strict=False)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# def face_shifter(Xt_raw, Xs_raw):
#     try:
#         Xs = detector.align(Xs_raw, crop_size=(256, 256))
#     except Exception as e:
#         print('the target image is wrong, please change the image')
#     Xs = test_transform(Xs)
#     Xs = Xs.unsqueeze(0).cuda()
#
#     with torch.no_grad():
#         embeds = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
#
#     try:
#         Xt, trans_inv = detector.align(Xt_raw, crop_size=(256, 256), return_trans_inv=True)
#     except Exception as e:
#         return transform2(Xt_raw)
#     Xt_raw = np.array(Xt_raw)[:, :, ::-1]
#     Xt_raw = Xt_raw.astype(float) / 255.0
#     Xt = test_transform(Xt)
#     Xt = Xt.unsqueeze(0).cuda()
#
#     mask = np.zeros([256, 256], dtype=float)
#     for i in range(256):
#         for j in range(256):
#             dist = np.sqrt((i - 128) ** 2 + (j - 128) ** 2) / 128
#             dist = np.minimum(dist, 1)
#             mask[i, j] = 1 - dist
#     mask = cv2.dilate(mask, None, iterations=20)
#
#     with torch.no_grad():
#         Yt, _ = G(Xt, embeds)
#         Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5
#         Yt = Yt[:, :, ::-1]
#         Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
#         mask_ = cv2.warpAffine(mask, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
#         mask_ = np.expand_dims(mask_, 2)
#         Yt_trans_inv = mask_ * Yt_trans_inv + (1 - mask_) * Xt_raw
#         Yt_trans_inv = Yt_trans_inv * 255
#         Yt_trans_inv = Yt_trans_inv[:, :, ::-1]
#         shifter_tensor = transform2(Yt_trans_inv)
#         return shifter_tensor
#
#
# def face_shifter_batch(source_batch, target_batch):
#     output_batch = source_batch.clone()
#     for i in range(source_batch.size(0)):
#         source = source_batch[i] / 2.0 + 0.5
#         img_source = transforms.ToPILImage()(source)
#         target = target_batch[i] / 2.0 + 0.5
#         img_target = transforms.ToPILImage()(target)
#
#         output_batch[i] = face_shifter(img_source, img_target)
#
#     return output_batch


def face_shifter(xt, xs):
    _, _, w, h = xt.shape
    with torch.no_grad():
        embeds = arcface(F.interpolate(xs, (112, 112), mode='bilinear', align_corners=True))
        yt, _ = G(F.interpolate(xt, (256, 256), mode='bilinear'), embeds)
        return F.interpolate(yt, (w, h), mode='bilinear')


if __name__ == "__main__":
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    img_target = 'images/celeba_aligned_224_sample.jpg'
    img_source = 'images/celeba_aligned_224_sample2.jpg'
    img_tensor_s = transform2(Image.open(img_source))
    img_tensor_batch_s = img_tensor_s.repeat(8, 1, 1, 1).to(device)
    img_tensor_t = transform2(Image.open(img_target))
    img_tensor_batch_t = img_tensor_t.repeat(8, 1, 1, 1).to(device)
    face_shifter_tensor = face_shifter(img_tensor_batch_s, img_tensor_batch_t)
    save_image((face_shifter_tensor + 1.0) / 2.0, 'images/result.jpg', nrow=4)

