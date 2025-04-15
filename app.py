from flask import Flask, request, jsonify
from flask.helpers import send_file
import numpy as np
import onnxruntime
import cv2
import json

app = Flask(__name__,
            static_url_path='/', 
            static_folder='web')


session_large = onnxruntime.InferenceSession("efficientnet-lite4-11.onnx")
session_small = onnxruntime.InferenceSession("efficientnet-lite4-11-int8.onnx")


labels = json.load(open("labels_map.txt", "r"))


def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

@app.route("/")
def indexPage():
    return send_file("web/index.html")    

@app.route("/analyze", methods=["POST"])
def analyze():
    # Bild laden
    content = request.files.get('0', '').read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pre_process_edgetpu(img, (224, 224, 3))
    img_batch = np.expand_dims(img, axis=0)

    results_large = session_large.run(["Softmax:0"], {"images:0": img_batch})[0]
    top_large = reversed(results_large[0].argsort()[-5:])
    result_list_large = [{"class": labels[str(r)], "value": float(results_large[0][r])} for r in top_large]

    results_small = session_small.run(["Softmax:0"], {"images:0": img_batch})[0]
    top_small = reversed(results_small[0].argsort()[-5:])
    result_list_small = [{"class": labels[str(r)], "value": float(results_small[0][r])} for r in top_small]

  
    return jsonify({
        "original_model": result_list_large,
        "lite_model": result_list_small
    })
if __name__ == "__main__":
    app.run(debug=True)