
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from train import yolo_body, get_anchors, create_model, get_classes
from yolo import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import tensorflow.compat.v1 as tf
# ---------------------------config---------------------------
classes_path = 'model_data/pcb_classes.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors_path = 'model_data/yolo_anchors.txt'
anchors = get_anchors(anchors_path)

yolo = YOLO(model_path='yolo3-pcb1000-Missing.h5',
            classes_path=classes_path,
            anchors_path=anchors_path)
#--------------------------------------------------------------

def detect(file:str):
    path = file
    print(path)
    image = Image.open(path)
    r_image = yolo.detect_image(image)
    # 顯示圖片, r_image.show()
    r_image.save("detected.png")
    r_image.show()

if __name__ == "__main__":
    detect('output.png')