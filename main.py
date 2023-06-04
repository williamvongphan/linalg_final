# Image manipulation server for frontend.
import json
import os

import numpy as np
from flask import Flask, request, jsonify, render_template
from util.image import splitRGB, grayscale, mergeRGB
from reconstruct.lsq import lsq_reconstruct
from reconstruct.lasso import lasso_reconstruct
import base64
import requests
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1024 MB

@app.before_request
def handle_chunking():
    """
    Sets the "wsgi.input_terminated" environment flag, thus enabling
    Werkzeug to pass chunked requests as streams.  The gunicorn server
    should set this, but it's not yet been implemented.
    """

    transfer_encoding = request.headers.get("Transfer-Encoding", None)
    if transfer_encoding == u"chunked":
        request.environ["wsgi.input_terminated"] = True

import dotenv
dotenv.load_dotenv()

clipdrop_api_key = os.getenv('CLIPDROP_API_KEY')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lsq')
def lsq():
    return render_template('lsq.html')

@app.route('/lasso')
def lasso():
    return render_template('lasso.html')

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

@app.route('/reconstruct/ai', methods=['POST'])
def reconstruct_ai_route():
    # We assume that this request contains three entries: the original image, the mask image (black and white), and the prompt.

    headers = {
        'x-api-key': clipdrop_api_key
    }

    print(request.files)

    # Data should be multipart/form-data with two fields: image_file and mask_file
    data = {
        'image_file': (request.files.get('image_file').filename, request.files.get('image_file').stream, request.files.get('image_file').content_type),
        'mask_file': (request.files.get('mask_file').filename, request.files.get('mask_file').stream, request.files.get('mask_file').content_type),
    }

    # Send request
    response = requests.post('https://clipdrop-api.co/cleanup/v1', headers=headers, files=data)

    # Save response to file
    with open('response.png', 'wb') as f:
        f.write(response.content)

    # Convert the response to a base64 string (image/png)
    reso = base64.b64encode(response.content)

    # Response is the image, so send that image back to the frontend. Specify that it is an image/png.
    return reso, 200, {'Content-Type': 'image/png'}


@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'OK'})


if __name__ == '__main__':
    app.run(debug=True)