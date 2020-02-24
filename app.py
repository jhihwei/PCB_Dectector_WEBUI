import re
import numpy
import cv2
from flask import Flask, render_template, request, send_file, make_response
import base64
from PIL import Image
import time
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
    print(path)
    image = Image.open(path)
    r_image = yolo.detect_image(image)
    # 顯示圖片, r_image.show()
    r_image.save("detected.png")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/image', methods=['POST'])
def upload_image():
    npimg = numpy.fromfile(request.files['image'], numpy.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite('output.png', img)
    # time.sleep(5)
    keras.backend.clear_session()
    detect('output.png')
    with open('output.png', 'rb') as f:
        image_string = base64.b64encode(f.read())
    return image_string

if __name__ == '__main__':
    app.run(debug=False, threaded = False)
