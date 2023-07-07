from flask import Blueprint, json, request
import dlib
import numpy as np
import os

from facedetect import faceUtil
from recieveutils.recieveutils import getInput

faceType_blueprint = Blueprint('faceType', __name__)


@faceType_blueprint.route('/faceType', methods=['POST'])
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
    # picture_path = os.path.join(parent_dir, r'picture\bd2.jpg')
    #
    # # 读取图像文件
    # image = dlib.load_rgb_image(picture_path)

    # 检测人脸
    face_rectangles = face_detector(image)

    # 提取人脸特征向量
    face_embeddings = []
    known_embeddings = [[[-0.11100849509239197, 0.15551994740962982, 0.07666769623756409, -0.05591845512390137, -0.08876696228981018, 0.017641747370362282, -0.08633317798376083, -0.08121933788061142, 0.030130809172987938, 0.0009251069277524948, 0.18241697549819946, -0.011631221510469913, -0.2377934455871582, 0.053295135498046875, 0.0479716882109642, 0.12429185956716537, -0.0924975648522377, -0.048323556780815125, -0.24981525540351868, -0.0315714105963707, 0.016010325402021408, 0.014168426394462585, 0.06776497513055801, -0.025489619001746178, -0.15582013130187988, -0.21790139377117157, -0.07439624518156052, -0.09787655621767044, -0.009445407427847385, -0.06195206567645073, 0.0027198921889066696, -0.060225244611501694, -0.22699034214019775, -0.06113512068986893, -0.03915758430957794, -0.03017401322722435, -0.01592322625219822, -0.0614301823079586, 0.12000004202127457, 0.031767766922712326, -0.18022824823856354, 0.08591204881668091, 0.024118714034557343, 0.20404720306396484, 0.27373531460762024, -0.008297502063214779, 0.05507468804717064, -0.07708173245191574, 0.07594718039035797, -0.18345089256763458, 0.03897953778505325, 0.07158858329057693, 0.18180567026138306, 0.09854324907064438, 0.06888686120510101, -0.0417679101228714, 0.08205614984035492, 0.20762421190738678, -0.20966742932796478, 0.1030208095908165, 0.059922266751527786, -0.02176099643111229, -0.02098231203854084, -0.08431918174028397, 0.1644711196422577, 0.14575690031051636, -0.08958861976861954, -0.16265758872032166, 0.13272292912006378, -0.06522925198078156, -0.10987930744886398, 0.04720853641629219, -0.16847850382328033, -0.15818163752555847, -0.3715861141681671, 0.007842518389225006, 0.25143733620643616, 0.0921650379896164, -0.2940753996372223, -0.06396999955177307, -0.05870043486356735, -0.019425135105848312, -0.006157717201858759, 0.04747963324189186, -0.051297836005687714, -0.10175920277833939, -0.02300965040922165, 0.012733332812786102, 0.2625337541103363, -0.08957584202289581, -0.009821410290896893, 0.23779508471488953, 0.03519623726606369, -0.13984429836273193, -0.018503736704587936, 0.11075205355882645, -0.10268896073102951, -0.05064672604203224, -0.10096998512744904, -0.014275209046900272, -0.009711840189993382, -0.11000421643257141, -0.03680858388543129, 0.12609077990055084, -0.19958028197288513, 0.14155930280685425, -0.020562373101711273, -0.011153111234307289, -0.03106280229985714, -0.009656806476414204, -0.042801011353731155, -0.04846478998661041, 0.17704616487026215, -0.16722017526626587, 0.19892844557762146, 0.255460262298584, -0.04173678532242775, -0.02621917799115181, 0.013921372592449188, 0.08445294201374054, -0.02266423963010311, 0.14980581402778625, -0.14999687671661377, -0.1147625595331192, 0.062373388558626175, -0.019579024985432625, -0.02751634269952774, 0.08743269741535187]]]
    known_faces = ["bd"]
    threshold = 0.85
    for face_rect in face_rectangles:
        landmarks = landmark_predictor(image, face_rect)
        face_embedding = face_encoder.compute_face_descriptor(image, landmarks)
        face_embeddings.append(np.array(face_embedding))

        similarities = []
        for known_embedding in known_embeddings:
            # 待比对的特征向量
            face_embedding_np = np.array(face_embedding)
            known_embedding_np = np.array(known_embedding)
            # 归一化处理
            known_embedding_np = known_embedding / np.linalg.norm(known_embedding)
            face_embedding_np = face_embedding_np / np.linalg.norm(face_embedding_np)
            # 计算余弦相似度
            similarity = np.dot(known_embedding_np, face_embedding_np)
            similarities.append(similarity)

            # 根据相似度排序，找到最相似的人脸
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        most_similar_face = known_faces[max_index]

        # 可以根据相似度阈值判断是否为同一个人
        if max_similarity > threshold:
            recognized_person = most_similar_face
        else:
            recognized_person = 'Unknown'

        # 打印识别结果
        print('Recognized Person: ', recognized_person)

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
