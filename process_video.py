#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : process_video.py
#   Author      : YunYang1994, karsil
#   Created date: 2019-10-01 16:01:23
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os
import sys

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_fish.pb"
video_path      = 1 #0: live cam, 1: video
num_classes     = 2
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

filepath = sys.argv[1]
if video_path != 0 and os.path.exists(filepath):
    video_path = os.path.abspath(filepath)
elif video_path != 0:
    print("Missing parameter (video source)")
    raise SystemExit
else:
    # nothing to do here
    print("using live cam")

with tf.Session(graph=graph) as sess:
    print("reading video at " + video_path)
    vid = cv2.VideoCapture(video_path)
   
    # get frame size for saving
    w = int(vid.get(3))
    h = int(vid.get(4))

    head, tail = os.path.split(video_path)
    outputfileName = "out_" + tail

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputfileName, fourcc, 29, (w,h))
   
    # frames
    i = 1

    while(vid.isOpened()):
        print("Processing frame ", i)
        ret, frame = vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if(i > 1):
                print("Done. Quitting...")
                break;
            else:
                raise ValueError("No image!", frame)
        i = i + 1
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        ##cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(result)
        #cv2.imshow("result", result)
        #if cv2.waitKey(1) & 0xFF == ord('q'): break

vid.release()
out.release()
#cv2.destroyAllWindows()


