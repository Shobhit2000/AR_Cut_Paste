import image_masking
import numpy as np
import cv2
from flask import Flask, jsonify, request, make_response

app = Flask(__name__)

@app.route('/')
def index():
    return "App Works!!"

@app.route('/transfer/', methods=['POST'])
def transfer():
    if request.method == 'POST':

        # get image from app in multipart format
        print("Posted file: {}".format(request.files['file']))
        file = request.files['file']
        img = cv2.imread(file)
        img_array = np.asarray(img)

        # get the processed image from the base code
        img_rgba = image_masking.main(img_array)

        return jsonify({"img_array": img_rgba})


if __name__ == "__main__":
    app.run(debug=True)
