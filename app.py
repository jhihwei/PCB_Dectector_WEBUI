import re
import numpy
import cv2
from flask import Flask, render_template, request, send_file, make_response, jsonify
import base64
from PIL import Image
import time
import uuid
import json
from os import listdir
from os.path import isfile, join
import importlib
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


from train import yolo_body, get_anchors, create_model, get_classes
from yolo import YOLO
import keras
import gc
from keras import backend as K
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
    file_name = path[file.index('/')+1:]
    file_detected = "images/detected_"+file_name
    image = Image.open(path)
    r_image, info = yolo.detect_image(image)
    # 顯示圖片, r_image.show()
    r_image.save(file_detected)
    return file_detected, info

def get_sample_images_list():
    sample_path = 'static/samples'
    files = listdir(sample_path)
    files = [f for f in files if isfile(join(sample_path, f))]
    return files

@app.route('/', methods=['GET'])
def index():
    samples = get_sample_images_list()
    return render_template('index.html', samples=samples)

@app.route('/test', methods=['GET'])
def index_test():
    return render_template('index_test.html')


@app.route('/image', methods=['POST'])
def upload_image():
    data = {}
    file_name = "images/"+uuid.uuid4().urn[12:]+".jpg"
    npimg = numpy.fromfile(request.files['image'], numpy.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite(file_name, img)
    # keras.backend.clear_session()
    file_detected, info = detect(file_name)
    with open(file_detected, 'rb') as f:
        image_string = base64.b64encode(f.read())
    data['image_string'] = str(image_string,encoding='utf-8')
    data['info'] = info
    return json.dumps(data,ensure_ascii=False)

@app.route('/images', methods=['POST'])
def upload_local_image():
    file_name = 'static/samples/'+request.form['sample']
    file_detected, info = detect(file_name)
    with open(file_detected, 'rb') as f:
        image_string = base64.b64encode(f.read())
    data = {}
    data['image_string'] = str(image_string,encoding='utf-8')
    data['info'] = info
    return json.dumps(data,ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True, threaded = True, port=5002)
