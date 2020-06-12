"""**Import**"""

import tensorflow as tf

from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Declare classes which we want our model to classify and create a mask for
# define 81 classes that the coco model knowns about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

# Define the model and load coco model weights
def model():
    rcnn = MaskRCNN(mode='inference', model_dir=os.getcwd(), config=TestConfig())
    rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)

    return rcnn

# function which creates sticker
def get_sticker(img, mask):

  img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

  channel, _, _ = cv2.split(img)

  alpha_channel = np.ones(channel.shape, dtype=channel.dtype) * 255
  alpha_channel = alpha_channel * mask[:, :, 0]

  img_rgba[:, :, 3] = alpha_channel

  return img_rgba

def main(img):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rcnn = model()
    results = rcnn.detect([img], verbose=0)      # Make Prediction
    r = results[0]                               # Get dictionary for first prediction

    # display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']) # display image with bounding box, prediction class, prediction score and segmentation mask

    mask = r['masks']
    mask = mask.astype(int)

    img_rgba = get_sticker(img, mask)
    # img_rgba.save('obj_sticker.png')

    # plt.imshow(img_rgba)
    # plt.show()
    return img_rgba

if __name__ == "__main__":
    # img = cv2.imread("C://Users//SHOBHIT//Downloads//dog.jpg")
    # img = np.asarray(img)
    img = []
    _ = main(img)
