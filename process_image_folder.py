#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : pyCharm
#   File name   : process_image_folder.py
#   Author      : YunYang1994, karsil
#   Created date: 2019-12-17 15:10:14
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import sys
import os
import argparse
from datetime import datetime

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_fish.pb"
num_classes     = 2
input_size      = 416
graph           = tf.Graph()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Provide the path to a file with data')
    args = parser.parse_args()

    inputFilename = args.file_path

    if not os.path.exists(inputFilename):
        print(f"The folder {inputFilename} does not exist. Quitting...")
        sys.exit()


    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    sourcePathAbs = os.path.abspath(inputFilename)
    sourceFileHead, sourceFileTail = os.path.split(sourcePathAbs)
    outputPath = sourceFileTail + "_" + now
    targetFolder = sourceFileHead + "/" + outputPath
    print("Processed files will be saved to folder ", targetFolder)

    try:
        os.mkdir(targetFolder)
        print("Directory ", targetFolder, " created")
    except FileExistsError:
        print("Directory ", targetFolder, " already exists...")

    with open(inputFilename, 'r') as f:
        lines = f.readlines()

    with tf.Session(graph=graph) as sess:
        for line in lines:
            process(line.split()[0], targetFolder, sess)

def process(image_path, targetFolder, sess):
    head, tail = os.path.split(image_path)
    localFileName = tail

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
        feed_dict={return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)

    exportName = "out_" + localFileName

    filepath = targetFolder + "/" + exportName
    print("Done. Exporting image to ", filepath)

    image.save(filepath)

if __name__ == "__main__":
    main()
