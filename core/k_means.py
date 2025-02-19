# coding=utf-8
# k-means ++ for anchor generation
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import cfg

from kmeans_utils import kmeans, avg_iou

# Note: Call from project dir, not from inside core/

label_path = cfg.TRAIN.ANNOT_PATH
cluster_path = "./data/anchors/ufo_anchors.txt"
n_anchors = 9

def load_bboxes(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as f:
        txt = f.readlines()

    TUPEL_LEN = 5
    for line in txt:
        bbox_tupels = line.strip().split()[1:]
        if len(bbox_tupels) != 0:
            stripped = line.strip()
            assert len(stripped) % TUPEL_LEN == 0

            i = 0
            while i < len(stripped):
                annotations.append(stripped[i : i + TUPEL_LEN])
                i = i + TUPEL_LEN
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
    return np.array(box_sizes)

def sort_by_area(coords):
    def area(c):
        return c[0] * c[1]
    return np.array(sorted(coords, key=area))

def save_clusters_to_file(sorted_clusters, filepath):
    with open(filepath, "w") as f:
        for i, cluster in enumerate(sorted_clusters):
            clusterToString = str(int(cluster[0])) + "," + str(int(cluster[1]))
            if i+1 < len(sorted_clusters):
                clusterToString += ", "
            f.write(clusterToString)

def plot_clusters(clusters):
    largest_x = int(max(c[0] for c in clusters))
    largest_y = int(max(c[1] for c in clusters))
    offset = 50

    im = np.zeros([largest_y + 2 * offset,largest_x + 2 * offset, 3], dtype=np.uint8)
    im.fill(255)
    height, width, _ = im.shape

    fig,ax = plt.subplots(1)

    ax.set_title("Anchors of UFO data by K-means clustering based on IoU")
    ax.imshow(im)

    clusters = sort_by_area(clusters)

    print("small")
    for i, c in enumerate(clusters[:3]):
        print(f" {c[0]} / {c[1]}")
        x = width / 2 - (c[0]/2)
        y = height / 2 - (c[1]/2)
        rect = patches.Rectangle((x,y),c[0],c[1],linewidth=1,edgecolor='r',facecolor='none', label="Small" if i == 0 else "")
        ax.add_patch(rect)
        

    print("medium")
    for i, c in enumerate(clusters[3:6]):
        print(f" {c[0]} / {c[1]}")
        x = width / 2 - (c[0]/2)
        y = height / 2 - (c[1]/2)
        rect = patches.Rectangle((x,y),c[0],c[1],linewidth=1,edgecolor='b',facecolor='none', label="Medium" if i == 0 else "")
        ax.add_patch(rect)

    print("large")
    for i, c in enumerate(clusters[6:9]):
        print(f" {c[0]} / {c[1]}")
        x = width / 2 - (c[0]/2)
        y = height / 2 - (c[1]/2)
        rect = patches.Rectangle((x,y),c[0],c[1],linewidth=1,edgecolor='g',facecolor='none', label="Large" if i == 0 else "")
        ax.add_patch(rect)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
            fancybox=True, shadow=True, ncol=5)

    figure_path = "docs/anchors.png"
    print("Saving results to ", figure_path)

    plt.savefig(figure_path)
    plt.show()

annotations = load_bboxes(label_path)
boxes = []
for annotation in annotations:
    bboxes = parse_annotation(annotation)
    for box in bboxes:
        boxes.append(box)

boxes = get_pixel_differences_from_coords(boxes)
print(f"Got {len(boxes)} different bounding boxes, applying KMeans")

out = kmeans(boxes, k=n_anchors)
sorted_clusters = sort_by_area(out)
print("Accuracy: {:.6f}%".format(avg_iou(boxes, sorted_clusters)))
print("Boxes:\n {}".format(sorted_clusters))

ratios = np.around(sorted_clusters[:, 0] / sorted_clusters[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))

save_clusters_to_file(sorted_clusters, cluster_path)

plot_clusters(sorted_clusters)