import os

import dlib

from recieveutils.recieveutils import getModelPath


def loadFaceModel():
    # 加载人脸检测器
    face_detector = dlib.get_frontal_face_detector()

    # 获取模型文件的绝对路径
    model_dir = getModelPath()

    # 构造特征提取模型路径
    model_path = os.path.join(model_dir, 'shape_predictor_68_face_landmarks.dat')

    # 加载特征提取模型
    landmark_predictor = dlib.shape_predictor(model_path)

    # 构造人脸识别模型路径
    embedding_model_path = os.path.join(model_dir, 'dlib_face_recognition_resnet_model_v1.dat')

    # 加载人脸识别模型
    face_encoder = dlib.face_recognition_model_v1(embedding_model_path)

    return face_detector, landmark_predictor, face_encoder
