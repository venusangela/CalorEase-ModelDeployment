import os
import re
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
from collections import Counter

app = Flask(__name__)
app.config["ALLOWED_IMAGE_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

def allowed_image_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1] in app.config["ALLOWED_IMAGE_EXTENSIONS"]

model_cat = ["daging", "jajanan", "karbo", "lauk", "olahan_daging", "sayur"]
models = {}
category_indexs = {}
threshold = 0.5

def read_label_file(label_path):
    id_pattern = r'id:\s*(\d+)'
    display_name_pattern = r'display_name:\s*"([^"]*)"'
    with open(label_path, 'r') as file:
        pbtxt_content = file.read()
        ids = [int(i) for i in re.findall(id_pattern, pbtxt_content)]
        display_names = re.findall(display_name_pattern, pbtxt_content)
    result = {}
    for i in range(len(display_names)):
        result[ids[i]] = {'id': ids[i], 'name': display_names[i]}
    return result

for cat in model_cat:
    models[cat] = tf.saved_model.load(f"custom_model_lite\{cat}_efficientdet_d0\saved_model")
    category_indexs[cat] = read_label_file(f"custom_model_lite/{cat}_efficientdet_d0/food_label_map.pbtxt")

@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        image = request.files['image']
        if image and allowed_image_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # preprocess
            img_np = np.array(Image.open(image_path))
            input_tensor = tf.convert_to_tensor(img_np, dtype=tf.uint8)
            input_tensor = input_tensor[tf.newaxis, ...]
            input_tensor = input_tensor[:, :, :, :3]

            obj_detected = []
            for cat in model_cat:
                detections = models[cat]((input_tensor))
                classes = detections['detection_classes'][0].numpy()
                scores = detections['detection_scores'][0].numpy()
                for i in range(len(scores)):
                    if((scores[i] > threshold) and (scores[i] <= 1.0)):
                        object_name = category_indexs[cat][classes[i]]['name']
                        obj_detected.append(object_name)
            detections_dict = dict(Counter(obj_detected))

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success",
                },
                "data": {
                    "prediction": detections_dict,
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Bad request"
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

if __name__=="main":
    app.run()