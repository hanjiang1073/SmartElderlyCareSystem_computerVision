import os
import cv2
from flask import Blueprint, json, request

from recieveutils.recieveutils import getInput, getModelPath

banarea_blueprint = Blueprint('banarea', __name__)


@banarea_blueprint.route('/banarea', methods=['POST'])
def banareadetect():

    image = getInput(request)

    # 获取模型文件的绝对路径
    model_dir = getModelPath()

    cascade_path = os.path.join(model_dir, 'haarcascade_fullbody.xml')  # 行人检测器的权重文件路径
    cascade = cv2.CascadeClassifier(cascade_path)
    #image_path = r'D:\Desktop\git\SmartElderlyCareSystem_computerVision\picture\b1.jpg'  # 待检测的图像文件路径
    #image = cv2.imread(image_path)


    # 定义禁区的坐标范围
    x1, y1 = 150, 100  # 禁区左上角坐标
    x2, y2 = 400, 200  # 禁区右下角坐标



    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 行人检测
    pedestrians = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))

    # 判断行人是否踏入禁区
    for (x, y, w, h) in pedestrians:
        # 行人边界框的四个角点坐标
        x1_p, y1_p = x, y
        x2_p, y2_p = x + w, y + h

        # 判断行人边界框是否与禁区相交
        if x2_p >= x1 and x1_p <= x2 and y2_p >= y1 and y1_p <= y2:
            # 行人踏入禁区
            print("行人踏入禁区")
            cv2.putText(image, 'in', (x+10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制红色边框表示踏入禁区
        else:
            # 行人未踏入禁区
            print("行人未踏入禁区")
            cv2.putText(image, 'out', (x+10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制绿色边框表示未踏入禁区

    # 在图像上绘制禁区矩形框
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image, 'restricted zone', (x1+10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cascade = None

    # 显示结果图像
    cv2.imshow("Pedestrian Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

