import os
from deepface import DeepFace
from flask import Blueprint, json, request

from recieveutils.recieveutils import getFilePath, getModelPath

emotion_blueprint = Blueprint('emotion', __name__)


@emotion_blueprint.route('/emotion', methods=['POST'])
def emotiondetect():
    # 指定模型文件路径
    # 获取模型文件的绝对路径
    model_dir = getModelPath()

    # 构造特征提取模型路径
    model_path = os.path.join(model_dir, 'facial_expression_model_weights.h5')

    # 加载模型
    model = DeepFace.build_model('Emotion')

    # 加载模型权重
    model.load_weights(model_path)

    # 使用模型进行情绪识别
    # 图像路径
    image_path = getFilePath(request)

    # 进行情绪识别
    result = DeepFace.analyze(img_path=image_path, actions=['emotion'])

    # 提取情绪结果
    emotion_predictions = result[0]['emotion']
    emotion_label = max(emotion_predictions, key=emotion_predictions.get)

    # ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # 输出结果
    print("Predicted emotion:", emotion_label)
    model = None
    # 删除保存的图像文件
    os.remove(image_path)

    return json.dumps({'emotion_label': emotion_label})
