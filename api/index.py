from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import glob
import csv
import os
from pathlib import Path

# Get the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Initialize Flask app with correct template and static folders
app = Flask(__name__,
            template_folder=str(BASE_DIR / 'templates'),
            static_folder=str(BASE_DIR / 'static'))

# Configuration
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'static' / 'uploads')
app.config['DATASET_FOLDER'] = str(BASE_DIR / 'static' / 'dataset')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)

class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for (startX, endX, startY, endY) in segments:
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(ellipMask, cornerMask)
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        hist = self.histogram(image, ellipMask)
        features.extend(hist)
        return features

    def histogram(self, image, mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None, 255, 0, cv2.NORM_MINMAX).flatten()
        return hist

class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def search(self, queryFeatures, limit=10):
        results = {}
        if not os.path.exists(self.indexPath):
            return []

        with open(self.indexPath, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    features = [float(x) for x in row[1:]]
                    d = self.chi2_distance(features, queryFeatures)
                    results[row[0]] = d
                except ValueError:
                    continue

        results = sorted([(v, k) for (k, v) in results.items()])
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        return d

def index_images(dataset, index):
    cd = ColorDescriptor((8, 12, 3))
    with open(index, "a", newline='') as output:
        for imagePath in glob.glob(os.path.join(dataset, "*.jpg")):
            imageID = os.path.basename(imagePath)
            if imageID not in get_indexed_images(index):
                try:
                    image = cv2.imread(imagePath)
                    if image is not None:
                        features = cd.describe(image)
                        features = [str(f) for f in features]
                        output.write("%s,%s\n" % (imageID, ",".join(features)))
                    else:
                        print(f"Failed to load image: {imagePath}")
                except cv2.error as e:
                    print(f"OpenCV Error: {e}")

def get_indexed_images(index):
    indexed_images = set()
    if os.path.exists(index):
        with open(index, "r", newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    indexed_images.add(row[0])
    return indexed_images

def search_images(query, index, limit=10):
    cd = ColorDescriptor((8, 12, 3))
    queryImage = cv2.imread(query)
    if queryImage is None:
        return []
    queryFeatures = cd.describe(queryImage)
    searcher = Searcher(index)
    results = searcher.search(queryFeatures, limit=limit)
    return results

# Routes
@app.route('/')
def index():
    """Centralized index page"""
    return render_template('index.html')

@app.route('/home')
def home():
    """Default home page for the application"""
    return render_template('home.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        return render_template('search.html')

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        limit = int(request.form.get('limit', 10))
        index_path = str(BASE_DIR / 'index.csv')

        # Save uploaded file
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        # Search for similar images
        results = search_images(file_path, index_path, limit)

        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        return render_template('result.html', results=results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/indexing', methods=['GET', 'POST'])
def index_dataset():
    """Renamed from /index to /indexing"""
    if request.method == 'GET':
        return render_template('indexing.html')

    try:
        dataset_path = app.config['DATASET_FOLDER']
        index_path = str(BASE_DIR / 'index.csv')
        index_images(dataset_path, index_path)
        return jsonify({'message': 'Image indexing completed successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_index')
def update_index():
    try:
        dataset_path = app.config['DATASET_FOLDER']
        index_path = str(BASE_DIR / 'index.csv')
        cd = ColorDescriptor((8, 12, 3))
        indexed_images = get_indexed_images(index_path)

        for imagePath in glob.glob(os.path.join(dataset_path, "*.jpg")):
            imageID = os.path.basename(imagePath)
            if imageID not in indexed_images:
                image = cv2.imread(imagePath)
                if image is not None:
                    features = cd.describe(image)
                    features = [str(f) for f in features]
                    with open(index_path, "a", newline='') as output:
                        output.write("%s,%s\n" % (imageID, ",".join(features)))

        return jsonify({'message': 'Index update completed'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results():
    return render_template('result.html', results=[])

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
