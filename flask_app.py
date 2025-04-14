from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
from werkzeug.utils import secure_filename
from imageAnalyser import generate_material_predictions, visualise_material_regions
import io
from PIL import Image
import logging
import time
import glob

app = Flask(__name__)
UPLOAD_FOLDER = 'Uploads'
RESULT_FOLDER = 'Results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(filename='/home/fokusmok/flask_app.log', level=logging.DEBUG)

def allowed_file(filename):
    if not filename or '.' not in filename:
        app.logger.error(f"Invalid filename: '{filename}'")
        return False
    extension = filename.rsplit('.', 1)[-1].lower()
    is_allowed = extension in ALLOWED_EXTENSIONS
    app.logger.debug(f"Filename: '{filename}', Extension: '{extension}', Allowed: {is_allowed}")
    return is_allowed

@app.route('/')
def health_check():
    app.logger.info("Health check accessed")
    return jsonify({'status': 'ok'})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    app.logger.info("Analyze endpoint called")

    if 'image' not in request.files:
        app.logger.error("No image uploaded")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    app.logger.debug(f"Received file: '{file.filename}'")

    if file.filename == '':
        app.logger.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        app.logger.error(f"Invalid file type: '{file.filename}'")
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    app.logger.info(f"Saving file to {filepath}")
    file.save(filepath)

    try:
        # Analyze image
        app.logger.info("Generating material predictions")
        material_percentages = generate_material_predictions(filepath)

        # Generate unique filename for visualization
        timestamp = int(time.time())
        result_filename = f"result_{timestamp}_{filename}.png"
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        # Save visualization
        app.logger.info(f"Saving visualization to {result_filepath}")
        visualise_material_regions(filepath, material_percentages, result_filepath)

        # Clean up old visualization files (keep last 10)
        result_files = sorted(glob.glob(os.path.join(app.config['RESULT_FOLDER'], '*.png')), key=os.path.getmtime)
        while len(result_files) > 10:
            old_file = result_files.pop(0)
            app.logger.info(f"Removing old visualization: {old_file}")
            os.remove(old_file)

        # Prepare response
        response = {
            'report': {
                'Cotton': f"{material_percentages['Cotton']:.2f}%",
                'Leather': f"{material_percentages['Leather']:.2f}%"
            },
            'message': "Image analyzed successfully. See material breakdown above.",
            'visualization': f"/results/{result_filename}"
        }

        app.logger.info("Returning successful response")
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': f"Analysis failed: {str(e)}"}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            app.logger.info(f"Cleaning up {filepath}")
            os.remove(filepath)

@app.route('/results/<filename>')
def serve_result(filename):
    app.logger.info(f"Serving result: {filename}")
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/png')

@app.route('/fokusmok/activities.html')
def serve_static(filename):
    return send_from_directory('/home/fokusmok', filename)

if __name__ == '__main__':
    app.run()
