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

logdir = "./data/log/"

class YoloTrain(object):
    def __init__(self, stage):
        self.stage               = stage
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./data/log/train"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)
        self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.output_dir          = cfg.TRAIN.OUTPUT_FOLDER
        self.stage_1_ckpt        = cfg.TRAIN.STAGE_1_WEIGHT 
        self.stage_2_ckpt        = cfg.TRAIN.STAGE_2_WEIGHT 

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable    = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')

            if self.stage == 1:
                train_steps = tf.constant( self.first_stage_epochs * self.steps_per_period, dtype=tf.float64, name='train_steps')
            else:
                # for stage 2
                train_steps = tf.constant( self.second_stage_epochs * self.steps_per_period, dtype=tf.float64, name='train_steps')

            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)

            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=30)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",      self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            self.write_op = tf.summary.merge_all()
            self.summary_writer_train  = tf.summary.FileWriter(logdir, graph=self.sess.graph)

            self.logdir_eval = os.path.join(logdir, "eval")
            os.mkdir(self.logdir_eval)
            self.summary_writer_eval = tf.summary.FileWriter(self.logdir_eval, graph=self.sess.graph)

    def train_first_stage(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
            print('=> Success! Now it starts to train YOLOV3 from checkpoint...')
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')

        print("Training YOLOv3 - First stage")

        # Train first stage and keep track of best checkpoint
        best_performance_ckpt = ""
        best_performance_test_loss = float("inf")

        train_op = self.train_op_with_frozen_variables

        for epoch in range(1, 1 + self.first_stage_epochs):
            
            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []
            summaries = []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True,
                })

                train_epoch_loss.append(train_step_loss)
                print(summary)
                summaries.append(summary)
                pbar.set_description("train loss: %.2f" %train_step_loss)

            # WIP
            #mean_summary = np.mean(summaries)
            #summary = tf.Summary()
            #äsummary.value.add(tag="%sAccuracy" % prefix, simple_value=accuracy)
            #self.summary_writer_train.add_summary(mean_summary, epoch)
            # WIP End

            self.summary_writer_train.add_summary(summary, epoch)
            self.summary_writer_train.flush()

            pbar = tqdm(self.testset)
            for test_data in pbar:
                summary, test_step_loss = self.sess.run( [self.write_op, self.loss], feed_dict={
                                                self.input_data:   test_data[0],
                                                self.label_sbbox:  test_data[1],
                                                self.label_mbbox:  test_data[2],
                                                self.label_lbbox:  test_data[3],
                                                self.true_sbboxes: test_data[4],
                                                self.true_mbboxes: test_data[5],
                                                self.true_lbboxes: test_data[6],
                                                self.trainable:    False,
                })

                test_epoch_loss.append(test_step_loss)
                pbar.set_description("test loss: %.2f" %test_step_loss)

            self.summary_writer_eval.add_summary(summary, epoch)
            self.summary_writer_eval.flush()
            
            train_epoch_loss = np.mean(train_epoch_loss)
            test_epoch_loss = np.mean(test_epoch_loss)
            
            ckpt_file = self.stage_1_ckpt + "_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            
            self.saver.save(self.sess, ckpt_file, global_step=epoch)
            if test_epoch_loss < best_performance_test_loss:
                best_performance_test_loss = test_epoch_loss
                best_performance_ckpt = ckpt_file + "-" + str(epoch)

        print(f"Done with first stage after {self.first_stage_epochs} epochs")
        print(f"Best test loss of {best_performance_test_loss} had checkpoint {best_performance_ckpt}")

    def train_second_stage(self, best_performance_ckpt):
        self.sess.run(tf.global_variables_initializer())

        print('=> Restoring weights from: %s ... ' % best_performance_ckpt)
        self.loader.restore(self.sess, best_performance_ckpt)

        # Train stage and keep track of best checkpoint
        best_performance_ckpt = ""
        best_performance_test_loss = float("inf")

        print("Training YOLOv3 - Second stage")

        train_op = self.train_op_with_all_variables

        for epoch in range(1, 1 + self.second_stage_epochs):
            
            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True,
                })

                train_epoch_loss.append(train_step_loss)
                pbar.set_description("train loss: %.2f" %train_step_loss)

            self.summary_writer_train.add_summary(summary, epoch)
            self.summary_writer_train.flush()

            pbar = tqdm(self.testset)
            for test_data in pbar:
                summary, test_step_loss = self.sess.run( [self.write_op, self.loss], feed_dict={
                                                self.input_data:   test_data[0],
                                                self.label_sbbox:  test_data[1],
                                                self.label_mbbox:  test_data[2],
                                                self.label_lbbox:  test_data[3],
                                                self.true_sbboxes: test_data[4],
                                                self.true_mbboxes: test_data[5],
                                                self.true_lbboxes: test_data[6],
                                                self.trainable:    False,
                })

                test_epoch_loss.append(test_step_loss)
                pbar.set_description("test loss: %.2f" %test_step_loss)

            self.summary_writer_eval.add_summary(summary, epoch)
            self.summary_writer_eval.flush()
            
            train_epoch_loss = np.mean(train_epoch_loss)
            test_epoch_loss = np.mean(test_epoch_loss)
            
            ckpt_file = self.stage_2_ckpt + "_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            
            self.saver.save(self.sess, ckpt_file, global_step=epoch)

            if test_epoch_loss < best_performance_test_loss:
                best_performance_test_loss = test_epoch_loss
                best_performance_ckpt = ckpt_file + "-" + str(epoch)

        print(f"Done with second stage after {self.second_stage_epochs} epochs")
        print(f"Best test loss of {best_performance_test_loss} had checkpoint {best_performance_ckpt}")

                            
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
