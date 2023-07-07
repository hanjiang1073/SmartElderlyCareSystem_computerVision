import os
import cv2
import numpy as np
from flask import Blueprint, json, request

from recieveutils.recieveutils import getInput, getModelPath

interaction_blueprint = Blueprint('interaction', __name__)


@interaction_blueprint.route('/interaction', methods=['POST'])
def interactiondetect():
    # 获取模型文件的绝对路径
    model_dir = getModelPath()

    # 构造特征提取模型路径
    yolo_weight = os.path.join(model_dir, 'yolov3.weights')
    yolo_cfg = os.path.join(model_dir, 'yolov3.cfg')
    coco_names = os.path.join(model_dir, 'coco.names')

    # 加载YOLO模型
    net = cv2.dnn.readNet(yolo_weight, yolo_cfg)

    # 加载类别标签
    classes = []
    with open(coco_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # 设置输入图像尺寸和输出阈值
    input_width = 416
    input_height = 416
    confidence_threshold = 0.5
    nms_threshold = 0.4

    # 加载图像
    image = getInput(request)
    height, width, _ = image.shape

    # 构建输入图像的blob
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (input_width, input_height), swapRB=True, crop=False)

    # 设置模型输入
    net.setInput(blob)

    # 运行前向传播
    output_layers_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers_names)

    # 解析输出和筛选人体检测结果
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold and class_id == 0:  # 类别为人体
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 应用非最大抑制（NMS）来筛选重叠的边界框
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # 绘制检测结果和判断互动
    if len(indices) >= 2:
        for i in indices:
            # i = i[0]
            x, y, w, h = boxes[i]
            label = f"Person {i + 1}"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 判断两个人是否互动
        distance_threshold = 200  # 距离阈值

        # 检查是否有足够的人体边界框
        if len(indices) >= 2:
            x1, y1, _, _ = boxes[indices[0]]
            x2, y2, _, _ = boxes[indices[1]]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if distance < distance_threshold:
                interaction_status = "yes"
            else:
                interaction_status = "no"
        else:
            interaction_status = "not enough person"

        # 显示互动状态
        cv2.putText(image, interaction_status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    net = None

    # 显示结果图像
    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
