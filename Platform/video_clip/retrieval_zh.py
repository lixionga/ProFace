import torch
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image
from PIL import Image
import os
import numpy as np
import time

image_dir = os.listdir('static/img')
pipeline_retrieval = pipeline(task=Tasks.multi_modal_embedding,
                                  model='damo/multi-modal_clip-vit-large-patch14_336_zh')
image_num = len(image_dir)

def sim_image_text_zh(str):

    images = [Image.open(f"static/img/{i}.jpg") for i in range(1, image_num + 1, 1)]

    text = [str]

    img_embedding = pipeline_retrieval.forward({'img': images})['img_embedding']

    text_embedding = pipeline_retrieval.forward({'text': text})['text_embedding']

    with torch.no_grad():
        logits_per_image = text_embedding @ (img_embedding / pipeline_retrieval.model.temperature).t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    probs = probs[0]
    indices = np.argpartition(probs, -4)[-4:]

    if int(indices[0]) == image_num:
        top1 = image_num
    else:
        top1 = int(indices[0]) + 1

    if int(indices[1]) == image_num:
        top2 = image_num
    else:
        top2 = int(indices[1]) + 1

    if int(indices[2]) == image_num:
        top3 = image_num
    else:
        top3 = int(indices[2]) + 1

    if int(indices[3]) == image_num:
        top4 = image_num
    else:
        top4 = int(indices[3]) + 1

    result = [top1, top2, top3, top4]
    return result




