# imageAnalyser.py
import io
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Feature extraction functions (unchanged for brevity)
def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_histogram, _ = np.histogram(lbp.ravel(), bins=59, density=True)
    return lbp_histogram

def extract_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()[:64]
    return hist

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    return features

def extract_edge_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges)

def extract_features_from_patch(patch):
    lbp_hist = extract_lbp_features(patch)[:59]
    color_hist = extract_color_features(patch)[:64]
    glcm_features = extract_glcm_features(patch)[:4]
    edge_feature = np.array([extract_edge_features(patch)])
    return np.concatenate([lbp_hist, color_hist, glcm_features, edge_feature])

# Train and save model (run once, e.g., in a separate script)
def train_model():
    print('Training..')
    material_df = pd.read_csv("Materials.csv")
    X_features_data = []
    y_labels = []
    for _, row in material_df.iterrows():
        img_path = row['Image']
        label = row['Label']
        image = cv2.imread(img_path)
        if image is None:
            continue
        features = extract_features_from_patch(image)
        X_features_data.append(features)
        y_labels.append(label)

    if not X_features_data:
        raise ValueError("No valid images found for training")

    X = np.array(X_features_data, dtype=np.float32)
    y = np.array(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    selector = SelectKBest(score_func=f_classif, k=30)
    X_train_reduced = selector.fit_transform(X_train_scaled, y_train)

    svm_model = SVC(kernel='rbf', probability=True, C=0.5, gamma=0.1, class_weight='balanced')
    svm_model.fit(X_train_reduced, y_train)

    # Save model, scaler, and selector
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(selector, 'selector.joblib')
    print('Training saved to joblib files.')

    return svm_model, scaler, selector

# Load pre-trained model
def load_model():
    if not os.path.exists('svm_model.joblib'):
        raise FileNotFoundError("Model not found. Run train_model() first.")
    svm_model = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    selector = joblib.load('selector.joblib')
    return svm_model, scaler, selector

# Generate material predictions for an image
def generate_material_predictions(image_path, patch_size=32):
    svm_model, scaler, selector = load_model()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image")

    height, width, _ = image.shape
    material_counts = {'Cotton': 0, 'Leather': 0}

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            features = extract_features_from_patch(patch)
            features_scaled = scaler.transform([features])
            features_reduced = selector.transform(features_scaled)
            prediction = svm_model.predict(features_reduced)[0]
            material_counts[prediction] += 1

    total_patches = sum(material_counts.values())
    if total_patches == 0:
        return {'Cotton': 0, 'Leather': 0}

    # Convert counts to percentages
    material_percentages = {
        material: (count / total_patches * 100) for material, count in material_counts.items()
    }
    return material_percentages

# Visualization (optional, for debugging or frontend display)
def visualise_material_regions(image_path, material_percentages, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title("Material Analysis")
    plt.text(10, 30, f"Cotton: {material_percentages['Cotton']:.2f}%", color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(10, 60, f"Leather: {material_percentages['Leather']:.2f}%", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis("off")

    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close()

def testPrint():
    return 'this is test print'
