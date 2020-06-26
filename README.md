# YOLOv3 Implementation with TF1

## Part 1. Introduction

Implementation of YOLO v3 object detector in Tensorflow. The full details are in [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).  In this project we cover several segments as follows:<br>
- [x] [YOLO v3 architecture](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/yolov3.py)
- [x] [Training tensorflow-yolov3 with GIOU loss function](https://giou.stanford.edu/)
- [x] Basic working demo
- [x] Training pipeline
- [x] Multi-scale training method
- [x] Compute VOC mAP

YOLO paper is quick hard to understand, along side that paper. This repo enables you to have a quick understanding of YOLO Algorithmn.

## Part 2. Quick start
1. Clone this file
```bashrc
$ git clone https://github.com/karsil/tensorflow-yolov3.git
```
2.  You are supposed  to install some dependencies before getting out hands with these codes.
```bashrc
$ cd tensorflow-yolov3
$ pip install -r ./docs/requirements.txt
```
1. Exporting loaded COCO weights as TF checkpoint(`yolov3_coco.ckpt`)
```bashrc
$ cd checkpoint
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ tar -xvf yolov3_coco.tar.gz
$ cd ..
$ python convert_weight.py
$ python freeze_graph.py
```
4. Then you will get some `.pb` files in the root path.,  and run the demo script
```bashrc
$ python image_demo.py
$ python video_demo.py # if use camera, set video_path = 0
```

## Part 3. Train your own dataset
Two files are required as follows:

- [`dataset.txt`](https://raw.githubusercontent.com/YunYang1994/tensorflow-yolov3/master/data/dataset/voc_train.txt): 

```
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
```

- [`class.names`](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/data/classes/coco.names):

```
person
bicycle
car
...
toothbrush
```

Then edit your `./core/config.py` to make some necessary configurations

```bashrc
__C.YOLO.CLASSES                = "./data/classes/dataset.names"
__C.TRAIN.ANNOT_PATH            = "./data/dataset/train_dataset.txt"
__C.TEST.ANNOT_PATH             = "./data/dataset/test_dataset.txt"
```
Here are two kinds of training method: 

##### (1) train from scratch:

```bashrc
$ python train.py
$ tensorboard --logdir ./data
```
##### (2) train from COCO weights(recommended):

```bashrc
$ cd checkpoint
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ tar -xvf yolov3_coco.tar.gz
$ cd ..
$ python convert_weight.py --train_from_coco
$ python train.py
```

#### how to test and evaluate it ?
```
$ python evaluate.py
$ cd mAP
$ python main.py -na
```

##### (3) Inference with trained model
All scripts inside the root folder beginning with `process_` are meant for inference which are currently:
```bashrc
1. process_image.py
2. process_image_folder.py
3. process_video.py
4. process_video_to_images.py
```
To improve performance during inference, all of those scripts require a frozen graph of the checkpoint.
To to this, change the paths in `freeze_graph.py` like in the following to your own needs:
```bashrc
pb_file = "./yolov3_weights.pb"
ckpt_file = "./checkpoint/yolov3_test_loss=0.8979.ckpt-30"
```

Afterwards run it:
```bashrc
python freeze_graph.py
```

Then make sure that the desired `process_` script has the correct parameters and the previous freezed graph path is correct:
```bashrc
pb_file         = "./yolov3_weights.pb"
num_classes     = 4
input_size      = 416
score_threshold = 0.3
```
When calling the script, check for required parameters, e.g.:

```bashrc
python process_image_folder.py ./imagepaths.txt
```
