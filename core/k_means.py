# coding=utf-8
# k-means ++ for anchor generation
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from config import cfg

# Note: Call from project dir, not from inside core/

label_path = cfg.TRAIN.ANNOT_PATH
cluster_path = "./data/anchors/ufo_anchors.txt"
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

def sort_by_area(coords):
    def area(c):
        return c[0] * c[1]
    return sorted(coords, key=area)

def get_clusters(boxes):
    k_means = KMeans(n_clusters=n_anchors, max_iter=iterations_num, tol=loss_convergence)
    k_means.fit(boxes)
    clusters = np.around(k_means.cluster_centers_)
    print(f"Got {str(n_anchors)} anchors, which are (rounded):")
    print(clusters)
    return clusters

def save_clusters_to_file(sorted_clusters, filepath):
    with open(filepath, "w") as f:
        for i, cluster in enumerate(sorted_clusters):
            clusterToString = str(int(cluster[0])) + "," + str(int(cluster[1]))
            if i+1 < len(clusters):
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

    ax.set_title("Anchors of UFO data by K-means clustering")
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

    plt.savefig("docs/anchors.png")
    plt.show()

annotations = load_bboxes(label_path)
boxes = []
for annotation in annotations:
    bboxes = parse_annotation(annotation)
    for box in bboxes:
        boxes.append(box)

boxes = get_pixel_differences_from_coords(boxes)
print(f"Got {len(boxes)} different bounding boxes, applying KMeans")

clusters = get_clusters(boxes)

sorted_clusters = sort_by_area(clusters)

save_clusters_to_file(sorted_clusters, cluster_path)

plot_clusters(sorted_clusters)