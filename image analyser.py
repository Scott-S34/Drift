# importing libraries
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.patches as mpatches
from collections import Counter
import seaborn as sns

test_image_path = "shirt.jpg"
#test_image_path = "Business Model Canvas.jpg"

test_image = cv2.imread(test_image_path)
#test_image2 = cv2.imread(test_image_path2)

#extracting features using lbp
def extract_lbp_features(image):
    #convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #define LBP paramaters
    radius = 2 #neighbourhood size
    n_points = 8 * radius  # number of LBP points

    #compute local binary pattern 
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    #compute LBP histogram normalised
    #lbp_histogram, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    lbp_histogram, _ = np.histogram(lbp.ravel(), bins=59, density=True)
    return lbp, lbp_histogram

#extract colour features off an mimage
def extract_color_features(image):
    #convert image to HSV colour space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #compute normalised colour histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

# obtain texture detail
def extract_glcm_features(image):
    #convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #compute contrast, energy, homogenity, correaltion
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels =256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return gray, [contrast, energy, homogeneity, correlation]

#identify edges
def extract_edge_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges)

def extract_edge_feature_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

#function to combine all feature functions
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    #extract features
    _, lbp_hist = extract_lbp_features(image)
    color_hist = extract_color_features(image)
    _, glcm_features = extract_glcm_features(image)
    edge_feature = np.mean(extract_edge_features(image))


    #convert all into numpy arrays and flatten to reduce sizes 
    lbp_hist = np.array(lbp_hist, dtype=np.float32).flatten()[:59] #limit to 59 bins
    color_hist = np.array(color_hist, dtype=np.float32).flatten()[:64] #limit to 64 bins
    glcm_features = np.array(glcm_features).flatten()[:4] #limit to 4 bins
    edge_feature = np.array([edge_feature], dtype=np.float32) #keep as single value
    
    #reshape edge_feature into 1D array
    #edge_feature_mean_array = np.array([np.mean(edge_feature)])
    
    print(f"Color Histogram Shape: {color_hist.shape}")
    print(f"Edge Feature Shape: {edge_feature.shape}")
    print(f"LBP Histogram Shape: {lbp_hist.shape}")
    print(f"GLCM Histogram Shape: {glcm_features.shape}")

    feature_vector = np.concatenate([lbp_hist, color_hist, glcm_features, edge_feature])
    print(f"Total features extracted: {feature_vector.shape[0]}" )
    print("\n")
    assert feature_vector.shape[0]==128,f"Feature mismatch! Got {feature_vector.shape[0]} features instead of 128"
    return feature_vector

#extact features for patches 
def extract_features_from_patch(patch):
    #extract features without reading an image file
    color_hist = extract_color_features(patch).ravel()[:64]
    _, lbp_hist = extract_lbp_features(patch)
    lbp_hist = lbp_hist[:59]
    _, glcm_features = extract_glcm_features(patch)
    glcm_features = glcm_features[:4]
    edge_feature = np.array([np.mean(extract_edge_feature_img(patch))])

    #print(f"Patch feature count: {len(features)}")
    return np.concatenate([color_hist, lbp_hist, glcm_features, edge_feature])

#loading the dataset
material_df = pd.read_csv("Materials.csv")
print(material_df.head())

#extract features for each image
X_features_data = []
y_labels = []
#iterating over the rows and load images
for index, row in material_df.iterrows():
    img_path = row['Image'] #image path
    label = row['Label'] #the images corresponding label
    #image = Image.open(img_path) # load the image
    features = extract_features(img_path)

    if features is not None:
        #X_features_data.append(features)
        #y_labels.append(label)
        X_features_data.append([img_path] + list(features))
        y_labels.append(label)

if X_features_data:
    print(f"Shape of first feature vector: {len(X_features_data[0])}")

#convert to dataframe
#columns = ["Image", "Label", "LBP_Histogram", "Colour", "GLCM_Contrast", "GLCM_Energy", "GLCM_Homogeneity", "GLCM_Correlation", "Edge data"]
#adjust column names dynamically
test_features = extract_features(material_df.iloc[0]["Image"])

#call each feature function to get the sizes
#lbp_feature = extract_lbp_features(test_image)


#generate dynamic column names based on feature sizes
columns = (["Image"] +
           [f"LBP_{i}" for i in range(59)] +
           [f"Color_{i}" for i in range(64)] +
           #["GLCM_Contrast", "GLCM_Energy", "GLCM_Homogeneity", "GLCM_Correlation"]+
           [f"GLCM_{i}" for i in range(4)] + 
           ["Edge_Feature"])

print(f"Number of columns names provided: {len(columns)}")

if len(X_features_data) != len(columns):
    print(f"Mismatch! {len(X_features_data[0]) - len(columns)}")
for i, feature in enumerate(X_features_data):
    if len(feature) != len(X_features_data[0]):
        print(f"Row {i} has {len(feature)} features insteado of {len(X_features_data[0])}")


#features_df = pd.DataFrame(X_features_data, columns=columns)
features_df = pd.DataFrame(X_features_data, columns=columns) #exclude label and Image columns
#add image and label features back to the dataset
if "Image" not in features_df.columns:
    features_df.insert(0, "Image", material_df["Image"])
if "Label" not in features_df.columns:
    features_df.insert(1, "Label", material_df["Label"])

#features_df.insert(0, "Image", material_df["Image"])
#features_df.insert(0, "Label", material_df["Label"])
features_df.to_csv("Materials_Updated.csv", index=False)
print("Feature extraction completed and saved to Updated Materials.csv")

#image.show()

#ensure there is data before proceeding
if len(X_features_data) == 0:
    raise ValueError("Error: No valid images found for training")

#drop image path feature as it is not numerical
X_features_data_num = features_df.drop(["Image", "Label"], axis=1)
print(X_features_data_num.head())
#replace NaN with 0
X_features_data_num = np.nan_to_num(X_features_data_num)

features = extract_features(test_image_path)
print(f"Extracted features shape: {features.shape}")

X = features_df.drop(columns=["Image", "Label"])
y = material_df["Label"]

#split dataset (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) #stratify to ensure both classes are represented in both sets
svm_model = SVC(kernel='rbf', probability=True, C=0.5, gamma=0.1, class_weight='balanced')

#convert to numpy array
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
#apply MinMaxScaler
scaler = MinMaxScaler()
#scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
#selector = VarianceThreshold(threshold=0)
selector = SelectKBest(score_func=f_classif, k=30)
X_train_reduced = selector.fit_transform(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
X_test_reduced = selector.transform(X_test_scaled)
print("X_train shape", X_train_reduced.shape)
print("X_test shape:", X_test_reduced.shape)


#svm_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_reduced, y_train)

#features_df = pd.DataFrame(X_features_data, columns = ["LBP_Data", "Color_Feature", "GLCM_Contrast", "GLCM_Energy", "GLCM_Homogeneity", "GLCM_Correlation", "Edge_data"])
#features_df['Label'] = y_labels

#function for prediciting a clothing material
def predict_material(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # extract features
    edge_image = extract_edge_feature_img(image)
    lbp_image, lbp_hist = extract_lbp_features(image)
    glcm_image, glcm_features = extract_glcm_features(image)
    
    plt.figure(figsize=(12, 4))

    #show original image
    plt.subplot(1, 4 ,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original shirt")

    #show edge detection image
    plt.subplot(1, 4, 2)
    plt.imshow(edge_image, cmap='gray')
    plt.title("Edge Detection")

    #show LBP image
    plt.subplot(1, 4, 3)
    plt.imshow(lbp_image, cmap='gray')
    plt.title("Local Binary Pattern (LBP)")

    plt.subplot(1, 4, 4)
    plt.imshow(glcm_image)
    plt.title("GLCM Texture")

    # plt.subplot(1, 4, 5)
    # plt.imshow(color_image, cmap='grey')
    # plt.title("Colour Feature")

    plt.show()
    #return prediction[0]

#test model with a new image

predict_material(test_image_path)
#predicted_material = predict_material(test_image_path, scaler, svm_model)
#print("Final Prediction: ", predicted_material)
#evaluate model
y_pred = svm_model.predict(X_test_reduced)
#display confusion matrix
print(confusion_matrix(y_test, y_pred))
ax = plt.figure(figsize=(10, 5))
ax = sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, cbar=True)
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Confusion matrix representeing Actual vs predicted for material samples with SVM")
plt.show()

print(classification_report(y_test, y_pred))
accuracy = svm_model.score(X_test_reduced, y_test)
#accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%\n")
#print(classification_report(y_test, y_pred, target_names=["Cotton", "Leather"]))

# #test model with a new image
# predicted_material = predict_material(test_image_path, scaler, svm_model, "Leather")
# print("Final Prediction: ", predicted_material)
# #evaluate model
# y_pred = svm_model.predict(X_test_scaled)
# #accuracy = svm_model.score(X_test_scaled, y_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

#cross-validation evaluation technique
print("Cross-validation: ")
#scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5) #5-fold cross validation
scores = cross_val_score(svm_model, X_train_reduced, y_train, cv=5) #5-fold cross validation
print(f"Cross-validated scores: {scores}")
print(f"Mean accuracy: {scores.mean()* 100:.2f}%\n")

#hyperparameter tuning evaluation
print("Hyperparameter Tuning")
param_grid = {'C': [0.1, 1,5, 10,50, 100, 500, 1000], 'kernel': ['linear', 'rbf'], 'probability':[True, False]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")
 
def generate_material_predictions(image_path, patch_size=32):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    material_predictions = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            features = extract_features_from_patch(patch)
            #print(f"Extracted features at ({x}, {y}):", features)
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            #extract features from the patch
            features = extract_features_from_patch(patch)
            #scale features using trained scaler
            features_scaled = scaler.transform([features])
            features_reduced = selector.transform([features])
            #print(f"Scaled features at ({x}, {y}):", features_scaled)
            #predict material
            prediction = svm_model.predict(features_reduced)

            print(f"Patch at ({x}, {y}) predicted as: {prediction}") 
            #store (X, y, material) in the list
            material_predictions.append((x, y, prediction))
    return material_predictions


def visualise_material_regions(image_path, material_predictions, patch_size=32):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        return
    #create overlay for material regions
    overlay = np.zeros_like(image, dtype=np.uint8)
    #define colours
    cotton_color = (0, 0, 255)
    leather_color = (255, 0, 0)

    #apply materials
    for(x, y, material) in material_predictions:
        if material[0] == "Leather":
            cv2.rectangle(overlay, (x, y), (x + patch_size, y + patch_size), leather_color, -1)
        else:
            cv2.rectangle(overlay, (x, y), (x+patch_size, y + patch_size), cotton_color, -1)    

    #blend overlay with original image
    alpha = 0.2
    output = cv2.addWeighted(image, 1, overlay, alpha, 0)
    #plot image with a legend
    plt.figure(figsize=(8, 6))
    plt.imshow(output)
    plt.title("Material Classification by SVM")

    #create legend patches
    leather_patch = mpatches.Patch(color=(1, 0, 0, 0.6), label="Leather")
    cotton_patch = mpatches.Patch(color=(0, 0, 1, 0.6), label="Cotton")
    
    plt.legend(handles=[leather_patch, cotton_patch], loc="lower center", ncol=2, fontsize=10, frameon=True)
    
    plt.axis("off")
    plt.show()

#generate predictions
material_predictions = generate_material_predictions(test_image_path)
#print(material_predictions)
#visualise results
visualise_material_regions(test_image_path, material_predictions)

print("Training class distribution:", Counter(y_train))
print("Testing class distribution: ",Counter(y_test))
print("Model support vectors:", svm_model.support_vectors_.shape)


print("Number of Support Vectors per Class: ", svm_model.n_support_)

# plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm')
# plt.scatter(svm_model.support_vectors_[:,0], svm_model.support_vectors_[:, 1], edgecolors='k', marker='o', facecolors='none', s=150)
# plt.title("SVM Decision Boundary with Support Vectors")
# plt.show()
#assign numeric values
label_mapping = {'Cotton': 0, 'Leather': 1}
y_train_numeric = y_train.map(label_mapping)
plt.scatter(X_train_reduced[:, 0], X_train_reduced[:, 6], c=y_train_numeric, cmap='coolwarm', edgecolors='k')
plt.colorbar(label="Material Type")
plt.show()

#print("Scaling", np.min(X_train_scaled, axis=0), np.max(X_train_scaled, axis=0))
#print(X_train.shape)
#print("Feature Variance: ", np.var(X_train_scaled, axis=0))

plt.hist(X_train_reduced.flatten(), bins=50)
plt.title("Feature Value Distribution")
plt.show()

