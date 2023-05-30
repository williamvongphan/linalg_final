# Image manipulation server for frontend.
import json

import numpy as np
from flask import Flask, request, jsonify, render_template
from util.image import splitRGB, grayscale, mergeRGB
from reconstruct.lsq import lsq_reconstruct
from reconstruct.lasso import lasso_reconstruct
import base64
app = Flask(__name__)

# Path: /splitRGB
# Split image into RGB channels, return as list of 3 matrices.
@app.route('/splitRGB', methods=['POST'])
def splitRGBServer():
    # Get image from request. Assume image is a 3D matrix.
    image = request.get_json()['image']
    # Split image into RGB channels.
    red, green, blue = splitRGB(image)
    # Return list of 3 matrices.
    return jsonify({'red': red.tolist(), 'green': green.tolist(), 'blue': blue.tolist()})

# Path: /grayscale
# Convert image to grayscale.
@app.route('/grayscale', methods=['POST'])
def grayscaleServer():
    # Get image from request. Assume image is a 3D matrix.
    image = request.get_json()['image']
    # Convert image to grayscale.
    gray = grayscale(image)
    # Return grayscale image.
    return jsonify({'gray': gray.tolist()})

# Path: /mergeRGB
# Convert image from 3 matrices to 1 matrix.
@app.route('/mergeRGB', methods=['POST'])
def mergeRGBServer():
    # Get RGB channels from request. Assume each channel is a 2D matrix.
    red = request.get_json()['red']
    green = request.get_json()['green']
    blue = request.get_json()['blue']
    # Convert image from 3 matrices to 1 matrix.
    image = mergeRGB(red, green, blue)
    # Return image matrix.
    return jsonify({'image': image.tolist()})

# there are 5 pages which we will send from templates. index.html, lsq.html, lasso.html, iterlasso.html, and ai.html
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lsq')
def lsq():
    return render_template('lsq.html')

@app.route('/lasso')
def lasso():
    return render_template('lasso.html')

@app.route('/iterlasso')
def iterlasso():
    return render_template('iterlasso.html')

@app.route('/ai')
def ai():
    return render_template('ai.html')

@app.route('/reconstruct/lsq', methods=['POST'])
def reconstruct_lsq_route():
    imgArray = request.get_json()['imgArray']
    regParam = request.get_json()['regParam']
    imgArray = base64.b64decode(imgArray)
    imgArray = json.loads(imgArray)
    res = lsq_reconstruct(imgArray, regParam=regParam)
    res = json.dumps(res.tolist())
    res_bytes = res.encode('utf-8')
    res = base64.b64encode(res_bytes)
    return jsonify({'imgArray': res.decode('utf-8')})

@app.route('/reconstruct/lasso', methods=['POST'])
def reconstruct_lasso_route():
    imgArray = request.get_json()['imgArray']
    alpha = request.get_json()['alpha']
    imgArray = base64.b64decode(imgArray)
    imgArray = json.loads(imgArray)
    res = lasso_reconstruct(imgArray, alpha=alpha)
    res = json.dumps(res.tolist())
    res_bytes = res.encode('utf-8')
    res = base64.b64encode(res_bytes)
    return jsonify({'imgArray': res.decode('utf-8')})


if __name__ == '__main__':
    app.run(debug=True)