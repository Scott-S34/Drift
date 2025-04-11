
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import io
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = '/home/fokusmok/cloth-scanner/uploads'
# RESULT_FOLDER = '/home/fokusmok/cloth-scanner/Results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return render_template('activities.html')

@app.route('/analyze', methods = ['GET', 'POST'])
def analyze_image():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['image']

        # Check if the file has a filename
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check if the file extension is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                # Save the uploaded file
                file.save(filepath)

                # Path for imageAnalyser.py
                analyser_path = '/home/fokusmok/imageAnalyser.py'
                try:
                    # Run the script with the image filepath as an argument
                    result = subprocess.run(
                        ['python3', script_path, filepath],
                        capture_output=True, text=True, check=True
                    )
                except subprocess.CalledProcessError as e:
                    return jsonify({
                        'error': f'Script execution failed: {e.stderr}'
                    }), 500
                except FileNotFoundError:
                    return jsonify({
                        'error': f'Script not found at {script_path}'
                    }), 500

                return jsonify({'message': f'File {filename} uploaded successfully', 'filename': filename})
            except Exception as e:
                return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

        return jsonify({'error': 'Invalid file type'}), 400

    # Handle GET request
    return render_template('activities.html')

if __name__ == '__main__':
    app.run(debug=True)
