# -*- coding:utf-8 -*-
"""
作者：cyd
日期：2024年11月23日
"""
from collections import namedtuple
from torch.utils import model_zoo
from retinaface.predict_single import Model
import torch
import os


# 定义一个namedtuple用于表示模型相关信息（和原函数中的用法类似）
model = namedtuple("model", ["url", "model"])

models = {
    "resnet50_2020-07-20": model(
        url="https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip",  # noqa: E501
        model=Model,
    )
}


def get_model_from_local(model_name: str, max_size: int, device: str = "cpu", weights_path: str = None) -> Model:
    """
    从本地加载权重来获取指定名称的模型

    :param model_name: 模型名称，例如 "resnet50_2020-07-20"
    :param max_size: 模型相关的最大尺寸参数（按原函数要求保留此参数）
    :param device: 模型运行的设备，默认为 "cpu"
    :param weights_path: 本地权重文件的路径，需用户指定实际路径
    :return: 加载了本地权重的Model实例
    """
    if model_name not in models:
        raise ValueError(f"Model name {model_name} not found in available models.")
    if weights_path is None:
        raise ValueError("weights_path must be provided for local weight loading.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file at {weights_path} does not exist.")

    # 实例化模型
    model_instance = models[model_name].model(max_size=max_size, device=device)

    # 加载本地权重
    state_dict = torch.load(weights_path, map_location=device)
    model_instance.load_state_dict(state_dict)

    return model_instance