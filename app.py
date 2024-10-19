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

        # Compute the saliency map
        success, saliencyMap = saliency.computeSaliency(image)

        if not success:
            return jsonify({"error": "Saliency computation failed."}), 500  # Internal server error

        # Convert the saliency map to 8-bit grayscale image
        saliencyMap = (saliencyMap * 255).astype("uint8")

        # Threshold the saliency map to get the binary mask
        _, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Apply the mask to the original image to get the cleaned-up version
        masked_image = cv2.bitwise_and(image, image, mask=threshMap)

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
