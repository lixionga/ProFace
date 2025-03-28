import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from .backbones import get_model
from .utils.utils_config import get_config

class IDLoss(nn.Module):
    def __init__(self,
                 ref_path=None,
                 face_model='r50',
                 face_dataset='glint360k',
                 ):
        super(IDLoss, self).__init__()

        cfg = get_config(f'functions/arcface_torch/configs/{face_dataset}_{face_model}.py')
        self.face_model = get_model(cfg.network, dropout=0.0,
                                    fp16=cfg.fp16, num_features=cfg.embedding_size)
        ckpt_path = f'functions/arcface_torch/checkpoints/{face_dataset}_{face_model}.pth'
        a, b = self.face_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        # print('loading face model:', a, b)
        print('build model:', face_dataset, face_model, ckpt_path)
        self.face_model.eval()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))

        self.to_tensor = torchvision.transforms.ToTensor()

        self.ref_path = "/workspace/ddgm/functions/arcface/land.png" if not ref_path else ref_path
        from PIL import Image
        # img - 参照图像
        img = Image.open(self.ref_path)
        image = img.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(image)
        img = img * 2 - 1
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        self.ref = img

    def change_ref_image(self, ref_path):
        from PIL import Image
        # img - 参照图像
        img = Image.open(ref_path)
        image = img.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(image)
        img = img * 2 - 1
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        self.ref = img

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.face_model(x)
        return x_feats

    def get_similarity(self, image1, image2):
        img1_feat = self.extract_feats(image1)
        img2_feat = self.extract_feats(image2)
        sim = F.cosine_similarity(img1_feat, img2_feat, dim=1)
        sim = torch.linalg.norm(sim)
        return sim

    def get_similarity_l1_l2(self, image1, image2):
        img1_feat = self.extract_feats(image1)
        img2_feat = self.extract_feats(image2)
        sim = F.cosine_similarity(img1_feat, img2_feat, dim=1)
        sim = torch.linalg.norm(sim)
        l1_loss = F.l1_loss(img1_feat, img2_feat)
        l2_loss = F.mse_loss(img1_feat, img2_feat)
        return sim, l1_loss, l2_loss

    def get_residual(self, image):
        img_feat = self.extract_feats(image)     # 生成图像的身份特征
        ref_feat = self.extract_feats(self.ref)  # 参照图像的身份特征
        return ref_feat - img_feat

    def get_sim_residual(self, image):
        img_feat = self.extract_feats(image)  # 生成图像的身份特征
        ref_feat = self.extract_feats(self.ref)  # 参照图像的身份特征
        sim = 1.0 - F.cosine_similarity(img_feat, ref_feat, dim=1)
        return sim

    def get_idis_div_residual(self, image, xt):
        n = image.size(0)
        device = image.device

        idis = torch.tensor(0.0).to(device)
        for i in range(n):
            img_feat = self.extract_feats(image[i].unsqueeze(0))  # 第i张生成图像的身份特征
            ref_feat = self.extract_feats(self.ref)  # 参照图像的身份特征
            loss = F.cosine_similarity(img_feat, ref_feat, dim=1)
            loss = torch.linalg.norm(loss)
            idis += loss

        div = torch.tensor(0.0).to(device)
        for i in range(n):
            for j in range(i + 1, n):
                feat_i = self.extract_feats(image[i].unsqueeze(0))  # 第i张生成图像的身份特征
                feat_j = self.extract_feats(image[j].unsqueeze(0))  # 第j张生成图像的身份特征
                loss = F.cosine_similarity(feat_i, feat_j, dim=1)
                loss = torch.linalg.norm(loss)
                if loss < 0:
                    loss = 0.0 * loss
                div += loss

        anony_loss = idis + div

        norm = torch.linalg.norm(anony_loss)
        ng = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        return ng
