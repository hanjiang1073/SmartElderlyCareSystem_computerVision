import numpy as np
from flask import Flask, jsonify, request
import cv2
import face_recognition


app = Flask(__name__)

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
