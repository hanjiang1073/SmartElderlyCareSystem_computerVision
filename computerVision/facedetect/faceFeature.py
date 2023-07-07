from flask import Blueprint, json, request
import dlib
import numpy as np
import os

from facedetect import faceUtil
from recieveutils.recieveutils import getInput

faceFeature_blueprint = Blueprint('faceFeature', __name__)


@faceFeature_blueprint.route('/faceFeature', methods=['POST'])
def faceTypeDiscrimination():
    image = getInput(request)


    # 加载人脸检测器
    # 加载特征提取模型
    # 加载人脸识别模型
    face_detector, landmark_predictor, face_encoder = faceUtil.loadFaceModel()

    # # 获取当前文件的绝对路径
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    #
    # # 构造图片路径
    # parent_dir = os.path.dirname(current_dir)
    # parent_dir = os.path.dirname(parent_dir)
    #
    # picture_path = os.path.join(parent_dir, r'picture\bd1.jpg')
    #
    # # 读取图像文件
    # image = dlib.load_rgb_image(picture_path)



    # 检测人脸
    face_rectangles = face_detector(image)

    # 提取人脸特征向量
    face_embeddings = []
    for face_rect in face_rectangles:
        landmarks = landmark_predictor(image, face_rect)
        face_embedding = face_encoder.compute_face_descriptor(image, landmarks)
        face_embeddings.append(np.array(face_embedding))

    # 将人脸特征向量转换为 Python 原生数据结构
    face_embeddings_list = [embedding.tolist() for embedding in face_embeddings]

    # 将 Python 数据结构转换为 JSON 格式
    face_embeddings_json = json.dumps(face_embeddings_list)

    # 输出 JSON 格式的人脸特征向量
    # print(face_embeddings_json)

    face_detector = None
    landmark_predictor = None
    face_encoder = None


    return face_embeddings_json
