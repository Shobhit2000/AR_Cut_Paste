import image_masking
import numpy as np
import cv2
from PIL import Image
from flask import Flask, jsonify, request, make_response

app = Flask(__name__)

@app.route('/')
def index():
    return "App Works!!"

@app.route('/transfer/', methods=['POST'])
def transfer():
    if request.method == 'POST':

        file = request.files['image']
        img = Image.open(file)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # print(img.shape)
        # cv2.imshow('win',img)
        # cv2.waitKey(0)
        # img_array = np.asarray(img)

        img_rgba = image_masking.main(img)

        return jsonify({"img_array": img_rgba})

if __name__ == "__main__":
    app.run(debug=False)
