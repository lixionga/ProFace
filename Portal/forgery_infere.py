import torch
from forgery.forgery_model import Detector
import cv2
from models.prepare_data import LandmarkModel
from models.align_face import align_img
import numpy as np
import os
from forgery import heatmap_generator
def cv2tensor(img):
    # 从 BGR 转为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为 (C, H, W) 格式
    img = np.transpose(img, (2, 0, 1))
    # 转为 PyTorch 张量，并添加一个批次维度
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    # 归一化到 [0, 1]
    img = img / 255.0
    return img

class ForgeryInference:
    def __init__(self):
        self.model = Detector()
        self.app = LandmarkModel(name='landmarks')
        self.app.prepare(ctx_id=0,det_thresh=0.5,det_size=(224,224),mode='None')

        self.model.load_state_dict(torch.load('/home/chenyidou/x_test/web/forgery/weights/EG_FF++(raw).tar',map_location=torch.device('cpu'))["model"])

        self.model.eval()
    
    def forward(self,img_path):
        image_name = os.path.basename(img_path).split('.')[0] + '_cropped.png'
        image_path = os.path.join("/home/chenyidou/x_test/web/forgery/cropped_results", image_name)
        image = cv2.imread(img_path)
        face = cv2.resize(image,(380,380))
        # landmark,_ = self.app.get(image)
        # face,_ = align_img(image,landmark,size=380)
        cv2.imwrite(image_path,face)
        with torch.no_grad():
            img = cv2tensor(face)
            pred=self.model(img).softmax(1)[:,1].cpu().data.numpy().tolist()
        if image_path is not None and os.path.exists(image_path):
            heatmap_generator.generate_cam_image('cpu',image_path,aug_smooth=False,eigen_smooth=False
                                                 ,method='gradcam++',output_dir='/home/chenyidou/x_test/web/static/out'
                                                 )
        else:
            print(f"Error: Invalid cropped image path: {image_path}")
        return pred,image_path
    

if __name__ == '__main__':
    model = ForgeryInference()

    # pred,path = model.forward('/Users/mac/代码/web/demo_file/gA11.jpg')

    # print(pred)
    # print('路径',path)

    image_folder = '/home/chenyidou/x_test/images/crop'
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # 检查文件是否为图片格式
            image_path = os.path.join(image_folder, filename)
            preds, path = model.forward(image_path)
            print(f"Prediction for {filename}: {preds}, Path: {path}")