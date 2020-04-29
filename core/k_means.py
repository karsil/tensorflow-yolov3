# coding=utf-8
# k-means ++ for anchor generation
import numpy as np
import os
from sklearn.cluster import KMeans
from config import cfg

label_path = cfg.YOLO.TRAIN.ANNOT_PATH
n_anchors = 9
loss_convergence = 1e-6
grid_size = 13
iterations_num = 100
plus = 0

def load_bboxes(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as f:
        txt = f.readlines()

    for line in txt:
        bbox_tupels = line.strip().split()[1:]
        if len(bbox_tupels) != 0:
            stripped = line.strip()
            annotations.append(stripped)
    return annotations

def parse_annotation(annotation):
    bboxes = [list(map(lambda x: int(float(x)), tupel.split(",")[:-1])) for tupel in annotation.split()[1:]]
    return bboxes    

def get_pixel_differences_from_coords(data):
    box_sizes = []
    for box in data:
        xmin, ymin, xmax, ymax = box
        x_size = (xmax - xmin) 
        y_size = (ymax - ymin) 
        box_sizes.append([x_size, y_size])
    return box_sizes

annotations = load_bboxes(label_path)
boxes = []
for annotation in annotations:
    bboxes = parse_annotation(annotation)
    for box in bboxes:
        boxes.append(box)

boxes = get_pixel_differences_from_coords(boxes)
print(f"Got {len(boxes)} different bounding boxes, applying KMeans")

k_means = KMeans(n_clusters=n_anchors, max_iter=iterations_num, tol=loss_convergence)
k_means.fit(boxes)
clusters = np.around(k_means.cluster_centers_)
print(f"Got {str(n_anchors)} anchors, which are (rounded):")
print(clusters)

# TODO: Sorting in small, medium, large

with open("../data/anchors/ufo_anchors.txt", "w") as f:
    print(len(clusters))
    for i, cluster in enumerate(clusters):
        print(i)
        clusterToString = str(int(cluster[0])) + "," + str(int(cluster[1]))
        if i+1 < len(clusters):
            clusterToString += ", "
        f.write(clusterToString)