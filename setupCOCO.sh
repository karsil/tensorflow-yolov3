# should be run in python environment

pip install -r ./docs/requirements.txt

cd checkpoint
wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
tar -xvf yolov3_coco.tar.gz
cd ..
python convert_weight.py
python freeze_graph.py
