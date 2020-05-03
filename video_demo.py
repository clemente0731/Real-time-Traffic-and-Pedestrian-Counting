# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Author      : Clemente420
#   Created date: 2019-11-14
#
# ================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode

video_path = "./vehicle.mp4"

writeVideo_flag = True
# video_path      = 0 # 调用本机物理摄像头
#num_classes     = 80
num_classes = 80
input_size = 416

input_layer = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
utils.load_weights(model, "./yolov3.weights")
model.summary()
vid = cv2.VideoCapture(video_path)


if writeVideo_flag:
    w = int(vid.get(3))
    h = int(vid.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("./output.mp4", fourcc, 24, (w, h))


frame_no = 0
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image!")

    prev_time = time.time()
    frame_size = frame.shape[:2]
    image_data = utils.image_preporcess(
        np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    frame_no += 1
    pred_bbox = model.predict_on_batch(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.25)
    bboxes = utils.nms(bboxes, 0.25, method='nms')
    # image = utils.draw_bbox(frame, bboxes)

    #########################################
    curr_time = time.time()
    exec_time_1 = curr_time - prev_time
    fps = 1/exec_time_1
    image = utils.video_draw_bbox(frame, bboxes, fps)
    ########################################
    # FPS LOG记录
    curr_time_2 = time.time()
    exec_time_2 = curr_time_2 - prev_time
    # print("yolo_timecost {} {}\n".format(frame_no,exec_time_2*1000)) # python video_demo.py | grep yolo_timecost >  yolo.log

    # result = np.asarray(image)
    # cv2.putText(result, text=fps, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=1, color=(255, 0, 0), thickness=2)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)

    #### 录制视频 ####
    # 注意 目录output必须存在
    video_writer.write(result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if writeVideo_flag:
    video_writer.release()
cv2.destroyAllWindows()
