import os

import cv2
import numpy as np


def getInput(request):
    # 获取到图片文件对象
    image_file = request.files['image']
    # 从图片文件对象中读取数据并解码为图像
    image_data = image_file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def getFilePath(request):
    image = request.files['image']

    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'deepface.jpg')
    image.save(image_path)
    return image_path


def getModelPath():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    parent_dir = os.path.join(parent_dir, 'model')
    return parent_dir
