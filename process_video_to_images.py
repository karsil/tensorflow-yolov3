#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : process_video_to_images.py
#   Author      : karsil
#   Created date: 2019-12-17 17:55:23
#   Description :
#
# ================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os
import sys
import argparse
from datetime import datetime
import srt
from tqdm import tqdm

return_elements  = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file          = "./yolov3_fish.pb"
num_classes      = 2
input_size       = 416
score_threshold  = 0.3
graph            = tf.Graph()
return_tensors   = utils.read_pb_return_tensors(graph, pb_file, return_elements)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Provide the path to a file with data')
    args = parser.parse_args()

    inputFile = args.file_path

    if not os.path.exists(inputFile):
        print(f"The video file {inputFile} does not exist. Quitting...")
        sys.exit()

    base = os.path.splitext(inputFile)[0]
    srtFile = base + '.srt'
    print(srtFile)
    if not os.path.exists(srtFile):
        print(f"The srt file {inputFile} does not exist. Quitting...")
        sys.exit()

    # Reading srt file
    with open(srtFile, 'r') as f:
        data = f.read()
    srt_generator = srt.parse(data)
    srtData = list(srt_generator)
    print(f"Constructed generator with {len(srtData)} entries")

    targetFolder = createTargetFolder(inputFile)

    run(inputFile, srtData, targetFolder)

    print(f"Done! files have been saved to folder ", targetFolder)

def run(inputFile, srtData, targetFolder):
    with tf.Session(graph=graph) as sess:
        print("Reading video at " + inputFile)
        vid = cv2.VideoCapture(inputFile)

        maxFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Begin processing of video with {maxFrames} frames...")

        # frames
        currFrame = 0

        # progressbar
        pbar = tqdm(total=maxFrames)
        
        while (vid.isOpened()):
            ret, frame = vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                if (currFrame > 0):
                    # something has been processed earlier
                    print("Done. Quitting...")
                    break
                else:
                    raise ValueError("Error while reading!", frame)


            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time = time.time()

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
            
            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, score_treshold)
            bboxes = utils.nms(bboxes, 0.45, method='nms')

            #  When detection has been observed, save image
            if (len(bboxes) > 0):
                image = utils.draw_bbox(frame, bboxes)

                # saving as image
                image = Image.fromarray(image)
                exportName = srtData[currFrame].content
                filepath = targetFolder + "/" + exportName + ".jpg"
                
                # Save logfile for image
                # Format: TopleftX, TopleftY, BottomRightX, BottomRightY, Class ID
                filepathLog = targetFolder + "/" + exportName + ".txt"
                with open(filepathLog, "w") as logfile:
                    for i, bbox in enumerate(bboxes):
                        coor = np.array(bbox[:4], dtype=np.int32)
                        class_ind = int(bbox[5])
                        logfile.write(str(coor[0]) + ", " + str(coor[1]) + ", " + str(coor[2]) + ", " + str(coor[3]) + ", " + str(class_ind) + "\n")

                image.save(filepath)

            pbar.update(1)
            currFrame = currFrame + 1

        vid.release()

# returns absolute path of newly created target folder
def createTargetFolder(inputFile):
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # Saving files into folder 'inputFilename_%Y-%m-%d_%H:%M:%S'
    sourcePathAbs = os.path.abspath(inputFile)
    sourceFileHead, sourceFileTail = os.path.split(sourcePathAbs)
    outputPath = sourceFileTail + "_" + now
    targetFolder = sourceFileHead + "/" + outputPath

    try:
        os.mkdir(targetFolder)
        print("Target directory ", targetFolder, " created")
    except FileExistsError:
        print("Target directory ", targetFolder, " already exists...")

    return targetFolder

if __name__ == "__main__":
    main()