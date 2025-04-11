from flask import Flask, request, jsonify, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from imageAnalyser import svm_model, scaler, selector, generate_material_predictions, visualise_material_regions
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import io
from PIL import Image

print("render-server.py file on server.")  # Log message to terminal
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'Results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
print("Waiting for an image upload..")  # Log message to terminal

@app.route('/')
def health_check():
    return jsonify({'status': 'ok'})
    
@app.route('/analyze', methods=['POST'])
def analyze_image():
    print("Image upload received!")  # Log message to terminal
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    # Save the image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Generate material predictions
        material_predictions = generate_material_predictions(filepath)
        
        # Count predictions for summary
        materials = [pred[2][0] for pred in material_predictions]  # Extract material names
        material_counts = dict(Counter(materials))
        
        # Generate visualization
        result_filename = f'result_{filename}'
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        # Temporarily save visualization
        visualise_material_regions(filepath, material_predictions)
        plt.savefig(result_filepath, format='png', bbox_inches='tight')
        plt.close()
        
        # Prepare response
        response = {
            'predictions': material_counts,  # e.g., {"Cotton": 50, "Leather": 20}
            'visualization': f'/results/{result_filename}'  # URL to fetch the image
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/results/<filename>')
def serve_result(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/png')

port = int(os.environ.get('PORT', 10000))
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
