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
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgba = image_masking.main(img)
        cv2.imwrite('sticker.png', img_rgba)

        #return jsonify({"img_array": img_rgba})
        return "DONE"

if __name__ == "__main__":
    app.run(debug=False)
