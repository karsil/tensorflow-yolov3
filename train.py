#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
import argparse
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg

from YoloTrain import YoloTrain

logdir = "./data/log/"
              
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stage", type=int,
                    help="Select the stage which shall be trained ('1' or '2')")
    parser.add_argument("-c", "--ckpt", type=str,
                    help="Select the ckpt file which shall be trained in the second stage!")
    args = parser.parse_args()

    stage = int(args.stage)
    first_stage_ckpt = args.ckpt

    if stage is 1:
        if os.path.exists(logdir): shutil.rmtree(logdir)
        os.mkdir(logdir)
        logdir = logdir + "stage1/"
        os.mkdir(logdir)
        YoloTrain(1).train_first_stage()
    elif stage is 2:
        logdir = logdir + "stage2/"
        if os.path.exists(logdir): shutil.rmtree(logdir)
        os.mkdir(logdir)
        assert first_stage_ckpt is not None, "You want to train for second stage, but no ckpt is given!"
        YoloTrain(2).train_second_stage(first_stage_ckpt)
    else:
        print(f"Stage {stage} not valid (must be 1 or 2), exiting...")
