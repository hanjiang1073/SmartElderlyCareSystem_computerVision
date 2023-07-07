import numpy as np
from flask import Flask, jsonify, request
import cv2
import face_recognition

from banarea.ban import banarea_blueprint
from emotiondetect.emotion import emotion_blueprint
from facedetect.faceType import faceType_blueprint
from facedetect.faceFeature import faceFeature_blueprint
from interactiondetect.interaction import interaction_blueprint

app = Flask(__name__)

# 在主应用程序中注册蓝图
app.register_blueprint(faceType_blueprint)
app.register_blueprint(faceFeature_blueprint)
app.register_blueprint(banarea_blueprint)
app.register_blueprint(interaction_blueprint)
app.register_blueprint(emotion_blueprint)


@app.route('/')
def hello():
    return 'Hello, Flask!'


@app.route('/recognize', methods=['POST'])
def recognize_faces():
    # 获取上传的图像文件
    file = request.files['image']

    # 将图像文件转换为 OpenCV 图像格式
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # 在图像中检测人脸
    face_locations = face_recognition.face_locations(img)

    # 返回人脸识别结果
    response = {
        'num_faces': len(face_locations),
        'face_locations': face_locations
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()
