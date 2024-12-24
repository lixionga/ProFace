import torch
import clip
from PIL import Image
import os
image_dir = os.listdir('./static/img')
image_num = len(image_dir)

def sim_image_text(str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    images = [preprocess(Image.open(f"./static/img/{i}.jpg")) for i in range(1, image_num+1, 1)]

    images = torch.stack(images).to(device)
    text = clip.tokenize(str).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    values, indices = similarity.topk(3)

    if int(indices[0][0]) == image_num:
        top1 = image_num
    else:
        top1 = int(indices[0][0])+1

    if int(indices[0][1]) == image_num:
        top2 = image_num
    else:
        top2 = int(indices[0][1]) + 1

    if int(indices[0][2]) == image_num:
        top3 = image_num
    else:
        top3 = int(indices[0][2]) + 1

    result = [top1, top2, top3]
    return result


