from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Enable CORS for requests from 'http://localhost:3000'
CORS(app, origins=['http://localhost:3000'])

@app.route('/api/v1/process-image', methods=['POST'])
def process_image():
    """
    This endpoint processes an image and returns the result.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400  # Bad request

        file = request.files['file'].read()
        npimg = np.frombuffer(file, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image data."}), 400

        # Initialize the static saliency spectral residual detector
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        # saliency = cv2.saliency.ObjectnessBING_create()

        # Compute the saliency map
        success, saliencyMap = saliency.computeSaliency(image)

        if not success:
            return jsonify({"error": "Saliency computation failed."}), 500  # Internal server error

        #V1
        # saliencyMap = (saliencyMap * 255).astype("uint8")
        # _, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # black_background = np.zeros_like(image)
        # masked_image = np.where(threshMap[:, :, np.newaxis] == 255, image, black_background)

        #V2
        saliencyMap = cv2.normalize(saliencyMap, None, 0.1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Ensure the saliency map has three channels
        if len(saliencyMap.shape) == 2 or saliencyMap.shape[2] == 1:
            saliencyMap = cv2.cvtColor(saliencyMap, cv2.COLOR_GRAY2BGR)
        image_float = image.astype("float32")
        masked_image = cv2.multiply(image_float, saliencyMap)
        masked_image = np.clip(masked_image, 5, 255).astype("uint8")
        


        # Convert the processed image to a format that can be sent back
        _, buffer = cv2.imencode('.png', masked_image)
        io_buf = io.BytesIO(buffer)

        # Send the processed image back as a response
        return send_file(io_buf, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Internal server error

# A simple route to check the health of the API
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify if the API is running.
    """
    return jsonify({"status": "API is running"}), 200  # OK

if __name__ == '__main__':
    app.run(debug=True)
