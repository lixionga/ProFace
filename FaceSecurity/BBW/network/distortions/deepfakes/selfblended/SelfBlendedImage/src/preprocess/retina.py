import torch
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import numpy as np

def retina_landmarks(model, image_tensor):
    # Convert the image tensor to numpy array
    image_np = np.transpose(image_tensor.numpy(), (1, 2, 0))

    # Perform face detection
    faces = model.predict_jsons(image_np)

    # 提取人脸关键点信息
    x0, y0, x1, y1 = faces[0]['bbox']  # 只取第一个检测到的人脸
    landmarks = np.array([[x0, y0], [x1, y1]] + faces[0]['landmarks'])

    return landmarks

if __name__ == '__main__':
    # Example usage
    import torch

    # Load RetinaFace model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    model.eval()

    # Example input tensor (RGB image)
    input_image_tensor = torch.randn(3, 224, 224)

    # Detect faces and get bounding boxes
    # bounding_boxes = detect_faces(model, input_image_tensor)

    # Print bounding boxes
    # print('Bounding boxes:', bounding_boxes)
