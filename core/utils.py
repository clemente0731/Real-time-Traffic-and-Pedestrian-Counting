# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Author      : Clemente420
#   Created date: 2019-11-14
#
# ================================================================

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg
from core.sort import *
import collections

tracker = Sort()
memory = {}
indexIDs_memory = dict.fromkeys(range(90000), 0)  # 生成200个键值对 值初始化为0的记忆字典
counter_dict = collections.OrderedDict()


####### coco class ############
"""
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]
"""
# 分类号 参考上面coco class序号进行过滤，如person是0 如果不需要过滤 请注释该变量
# specified_class_id_filter = 2

# 字体样式
font_style = cv2.FONT_HERSHEY_SIMPLEX


############### 直线两个点设置##################

# 近似垂直线
# line = [(680, 100), (265, 720)]
#line = [(812, 89), (692, 1078)]
# line = [(748, 218), (656, 756)]
# line = [(917, 0), (917, 1065)]
# line = [(922, 0), (922, 1065)]
# # 近似水平线

line = [(0, 530), (2100, 530)] # for vehicle.mp4
# line = [(0, 430), (2100, 430)] # for people.mp4


# 近似计算直线 垂直 还是水平
def assess_horizontal_or_vertical(line):
    hrz_difference = line[1][0]-line[0][0]
    vtc_difference = line[1][1]-line[0][1]
    squared_difference = hrz_difference ** 2 - vtc_difference ** 2
    if squared_difference >= 0:
        horizontal_true_vertical_false = True
    else:
        horizontal_true_vertical_false = False

    return horizontal_true_vertical_false


# 当前计数线垂直还是水平，如果情况特殊可以手动设值，horizontal(True) or vertical(False)
horizontal_True_vertical_False = assess_horizontal_or_vertical(line)


def load_weights(model, weights_file):
    """
    I agree that this code is very ugly, but I don’t know any better way of doing it.
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(
            wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h,  w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def intersect(A, B, C, D):  # ab是目标框中心的两个近点，cd是线段
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def video_draw_bbox(image, bboxes, fps, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    image_h, image_w, _ = image.shape

    #############################################################
    # 判断水平或垂直计数，然后draw line
    if horizontal_True_vertical_False:
        # 画水平直线
        cv2.line(image, line[0], line[1], (254, 196, 8),
                 thickness=2, lineType=cv2.LINE_AA)  # 参数要求整数 除法用//
    else:
        cv2.line(image, line[0], line[1], (254, 196, 8),
                 thickness=2, lineType=cv2.LINE_AA)  # 参数要求整数 除法用//

    num_classes = len(classes)

    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(1)
    random.shuffle(colors)
    random.seed(None)

    np.set_printoptions(precision=3)
    #dets =[box[:5] for box in bboxes]
    ####################### 这里可以筛选要跟踪的分类号 ########################
    try:
        dets = [box[:]
                for box in bboxes if box[5] == specified_class_id_filter]
    except NameError:
        dets = [box[:] for box in bboxes]
    else:
        pass

    ####################### 不筛选 ########################
    # dets =[box[:] for box in bboxes ]

    # [[418.737 257.742 508.801 282.232  0.708 , cls_id]]
    dets = np.asarray(dets)
    # print("det返回多帧的数据框{}".format(dets))
    # 传入 [[x_min, y_min, x_max, y_max, probability, cls_id],...]
    tracks = tracker.update(dets)
    # print("tracks返回多帧的数据框{}".format(tracks))
    # tracks返回多帧的数据框[[6.342e+02 2.173e+02 1.121e+03 5.257e+02 9.910e-01 8.000e+00 1.000e+00]]
    # track = [ x_min, y_min, x_max, y_max, probability, cls_id, tracks_num]

    global memory
    boxes = []
    indexIDs = []
    previous = memory.copy()

    if bboxes is None:
        target_num = 0
    else:
        target_num = len(bboxes)

    for track in tracks:
        boxes.append([track[0], track[1], track[2],
                      track[3], track[4], track[5]])
        # boxes x y  x2 y2 probability cls_id 不同bbox对象的信息
        indexIDs.append(int(track[6]))
        memory[indexIDs[-1]] = boxes[-1]  # 这里控制p1 到底是第几帧的目标中心点 这里取的是上一帧
    if len(boxes) > 0:
        i = int(0)
        p0 = 0
        p1 = 0
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            probability = box[4]
            cls_id = int(box[5])
            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            bbox_color = colors[cls_id]
            bbox_thick = int(0.7 * (image_h + image_w) / 500)
            fontScale = 0.5

            # 判断是否计数字典里是否存在该键，没有该键就赋值0 初始化
            # 字典是嵌套形式的 {ship:{"in":0,"out":0}}

            #############################################################
            # 对应船舶的计数字典
            global counter_dict
            # 如果不存在相应分类的键，则初始化键值对为0
            if classes[cls_id] not in counter_dict:
                counter_dict[classes[cls_id]] = {}
                counter_dict[classes[cls_id]]['up'] = 0
                counter_dict[classes[cls_id]]['down'] = 0
                counter_dict[classes[cls_id]]['left'] = 0
                counter_dict[classes[cls_id]]['right'] = 0
            else:
                pass

            #############################################################
            # 统计部分
            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))  # 最新的目标中心点
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))  # 前一帧的目标中心点
                # 轨迹中心
                cv2.line(image, p0, p1, (254, 196, 8), 2,
                         lineType=cv2.LINE_AA)  # 把这两个中心的点连接(相当于轨迹)
                cv2.putText(image, "{}".format(
                    indexIDs[i]), (p0[0]+5, p0[1]+5), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (244, 208, 0), 1, lineType=cv2.LINE_AA)
                global indexIDs_memory

                # 判断进出 p0是当前帧目标中心坐标，p1是前一帧的目标中心点数
                if intersect(p0, p1, line[0], line[1]) and indexIDs_memory[indexIDs[i]] != 1:
                    # 如果是横向统计
                    if horizontal_True_vertical_False:
                        if p0[1] < p1[1]:  # 最新点的y坐标小于 之前点的y坐标 在向上走
                            counter_dict[classes[cls_id]]['up'] += 1  # 字典是嵌套形式
                            indexIDs_memory[indexIDs[i]] = 1
                        elif p0[1] > p1[1]:  # 最新点的y坐标 大于 之前点的y坐标 在向下走
                            counter_dict[classes[cls_id]]['down'] += 1
                            indexIDs_memory[indexIDs[i]] = 1
                        else:
                            pass
                    # 如果是左右统计
                    else:
                        if p0[0] < p1[0]:  # 最新点的x坐标小于 之前点的x坐标 在向左走
                            counter_dict[classes[cls_id]
                                         ]['left'] += 1  # 字典是嵌套形式
                            indexIDs_memory[indexIDs[i]] = 1
                        elif p0[0] > p1[0]:  # 最新点的x坐标 大于 之前点的x坐标 在向右走
                            counter_dict[classes[cls_id]]['right'] += 1
                            indexIDs_memory[indexIDs[i]] = 1
                        else:
                            pass
            i += 1

            #############################################################
            # 检测框及相关属性描述
            # 检测框
            cv2.rectangle(image, (x, y), (w, h), bbox_color, bbox_thick)

            # text = "{}({}):score:{:.4f}".format(classes[cls_id],indexIDs[i],probability) # 打印 类型
            # 描述标记文字和文字框
            text = "{}:{:.2f}".format(classes[cls_id], probability)  # 打印 类型
            t_size = cv2.getTextSize(
                text, 0, fontScale, thickness=bbox_thick)[0]
            # 画分类处的文字框
            cv2.rectangle(
                image, (x, y), (x + t_size[0], y - t_size[1]-3), bbox_color, thickness=-1)  # 画文字框
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//3, lineType=cv2.LINE_AA)

    # print("现在检测到的目标的种类数{}".format(len(counter_dict)))
    # print("现在检测到的当前目标的数{}".format(len(bboxes)))

    ################################### 数据面板绘制 ########################################

    # 图层1 先绘制信息面板矩形，以保持透明底层
    num_recorded_class = len(counter_dict)
    alpha = 0.3
    image_h, image_w, _ = image.shape
    overlay = image.copy()  # img副本 以供填充覆盖
    cv2.rectangle(image, (0, 0), (image_w//3 - 60,
                                  num_recorded_class * 40 + 110), (32, 36, 46), thickness=-1)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)  # img副本叠加

    # 图层2 信息面板文字描述

    vertical_increment = 20
    vertical_correction = 20
    horizontal_increment = image_w // 5
    up_or_left_sum = 0
    down_or_right_sum = 0
    text_thickness = int(0.6 * (image_h + image_w) / 1000)
    font_scale = 0.5
    # sum 纵向坐标偏移量
    sum_increment = num_recorded_class * 20 + 40

    ################ 按目标分类的计数信息填入 ####################
    # counter_dict = { speedboat {'up': 0, 'down': 0, 'left': 0, 'right': 0}, ..., river_boat {'up': 0, 'down': 0, 'left': 0, 'right': 0} }
    for key, values in counter_dict.items():
        vertical_correction += vertical_increment  # 每次新的键值对 纵向坐标 ++vertical_increment
        # 检测物类别具体描述 type
        cv2.putText(image, " {}".format(key), (0, vertical_correction), font_style,
                    font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)

        # 计数线接近水平放置时
        if horizontal_True_vertical_False:
            # up内河计数
            cv2.putText(image, "{}".format(values['up']), (horizontal_increment, vertical_correction),
                        font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
            # down内河计数
            cv2.putText(image, "{}".format(values['down']), (horizontal_increment+50, vertical_correction),
                        font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
            # 上下 累计计数
            up_or_left_sum += values['up']
            down_or_right_sum += values['down']

        # 计数线近似垂直放置时
        else:
            # left内河计数
            cv2.putText(image, "{}".format(values['left']), (horizontal_increment, vertical_correction),
                        font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
            # right内河计数
            cv2.putText(image, "{}".format(values['right']), (horizontal_increment+50, vertical_correction),
                        font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
            # 左右 累计计数
            up_or_left_sum += values['left']
            down_or_right_sum += values['right']

    ################ 不分左右上下的计数信息面板 ####################
    # 左下角 累计值
    cv2.putText(image, " cumulative count", (0, sum_increment), font_style,
                font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
    # 左下角 当前值 current_targets
    cv2.putText(image, " target number", (0, sum_increment+20), font_style,
                font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
    # 左下角 FPS 当前值
    cv2.putText(image, " fps", (0, sum_increment+40), font_style, font_scale,
                (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
    # # 左下角 时间
    # cv2.putText(image, " time" ,(0, sum_increment+60), font_style, font_scale, (254,196,8), thickness = text_thickness, lineType=cv2.LINE_AA)
    # 左下角 主网络
    cv2.putText(image, " detector", (0, sum_increment+60), font_style,
                font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
    # 左下角 署名
    cv2.putText(image, " author", (0, sum_increment+80), font_style, font_scale,
                (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)

    # 左上角 船型
    cv2.putText(image, " type", (0, vertical_increment), font_style, font_scale,
                (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)

    if horizontal_True_vertical_False:
        # up_count
        cv2.putText(image, "up", (horizontal_increment, vertical_increment), font_style,
                    font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # down_count
        cv2.putText(image, "down", (horizontal_increment+50, vertical_increment), font_style,
                    font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # 目前up/down计数的 cumulative count 数字
        cv2.putText(image, "{}".format(up_or_left_sum), (horizontal_increment, sum_increment),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(image, "{}".format(down_or_right_sum), (horizontal_increment+50, sum_increment),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # 当前目标数 current_targets
        cv2.putText(image, "{}".format(target_num), (horizontal_increment, sum_increment + 20),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(image, "{}".format(target_num), (horizontal_increment + 50, sum_increment + 20),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # 当前FPS资料
        cv2.putText(image, "{:0.1f}".format(fps), (horizontal_increment, sum_increment + 40),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(image, "{:0.1f}".format(fps), (horizontal_increment + 50, sum_increment + 40),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # # 当前时间
        # cv2.putText(image,"{}".format( time.strftime("%Y%m%d %H:%M:%S", time.localtime()) ),(horizontal_increment , sum_increment + 60 ), font_style, font_scale, (254,196,8), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 主网络
        cv2.putText(image, "{}".format("YOLOv3"), (horizontal_increment, sum_increment + 60),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # 署名描述
        cv2.putText(image, "{}".format("Clemente420"), (horizontal_increment, sum_increment + 80),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
    else:
        # left_count
        cv2.putText(image, "left", (horizontal_increment, vertical_increment), font_style,
                    font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # right_count
        cv2.putText(image, "right", (horizontal_increment+50, vertical_increment), font_style,
                    font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # 目前left/right计数的 cumulative count 数字
        cv2.putText(image, "{}".format(up_or_left_sum), (horizontal_increment, sum_increment),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(image, "{}".format(down_or_right_sum), (horizontal_increment+50, sum_increment),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # 当前目标数 current_targets
        cv2.putText(image, "{}".format(target_num), (horizontal_increment, sum_increment + 20),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(image, "{}".format(target_num), (horizontal_increment + 50, sum_increment + 20),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # 当前FPS资料
        cv2.putText(image, "{:0.1f}".format(fps), (horizontal_increment, sum_increment + 40),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(image, "{:0.1f}".format(fps), (horizontal_increment + 50, sum_increment + 40),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # # 当前时间
        # cv2.putText(image,"{}".format( time.strftime("%Y%m%d %H:%M:%S", time.localtime()) ),(horizontal_increment , sum_increment + 60 ), font_style, font_scale, (254,196,8), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 主网络
        cv2.putText(image, "{}".format("YOLOv3"), (horizontal_increment, sum_increment + 60),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)
        # 署名描述
        cv2.putText(image, "{}".format("Clemente420"), (horizontal_increment, sum_increment + 80),
                    font_style, font_scale, (254, 196, 8), thickness=text_thickness, lineType=cv2.LINE_AA)

    return image


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 450)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(
                bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(
                image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or(
        (pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(
        pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and(
        (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
