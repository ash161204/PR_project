import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------------------
# CONFIG (GENERALIZED PATH)
# -----------------------------------------
DATASET_PATH = "bloodcells_dataset" 
IMAGE_SIZE = (128, 128)
MAX_IMAGES_PER_CLASS = 2000 
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
# --- NEW CONFIG ---
SAVE_DIR = "data" 
# --- FILENAME FOR CONSOLIDATED PICKLE ---
LE_FILENAME = "label_encoder.pkl" 

# -----------------------------------------
# FUNCTION: Extract features
# -----------------------------------------
def extract_features(img):
    """
    Extracts HOG and LBP features from an input image.
    """
    # Resize
    img = cv2.resize(img, IMAGE_SIZE)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG Features
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )

    # LBP Features
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    # Compute the histogram of the LBP descriptor
    (hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2)
    )
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # final feature vector = HOG + LBP
    features = np.hstack([hog_feat, hist])
    return features


# -----------------------------------------
# LOAD DATASET (LIMITED)
# -----------------------------------------
X = []
y = []

print("Loading dataset...")

# Check if the dataset path is valid
if not os.path.isdir(DATASET_PATH):
    print(f"Error: Dataset path not found: '{DATASET_PATH}'. Please ensure the dataset is in the correct location relative to the script.")
    exit()

for label_name in os.listdir(DATASET_PATH):

    class_dir = os.path.join(DATASET_PATH, label_name)
    if not os.path.isdir(class_dir):
        continue

    print(f"Processing class: {label_name}")

    count = 0

    for file in os.listdir(class_dir):

        if count >= MAX_IMAGES_PER_CLASS:
            break

        file_path = os.path.join(class_dir, file)

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not read image: {file_path}")
            continue

        try:
            features = extract_features(img)
            X.append(features)
            y.append(label_name)
            count += 1 
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue


X = np.array(X)
y = np.array(y)

print("\n--- Feature Extraction Summary ---")
print(f"Total images loaded: {len(X)}")
print("Feature matrix shape:", X.shape)
print("----------------------------------\n")


# -----------------------------------------
# LABEL ENCODING & TRAIN-TEST SPLIT
# -----------------------------------------
print("Applying Label Encoding...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Classes: {le.classes_}")

print("Splitting data into Train and Test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\n--- Split Shapes ---")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("--------------------\n")


# -----------------------------------------
# FEATURE SCALING (Standardization)
# -----------------------------------------
print("Applying Feature Scaling (Standardization)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Scaling complete.")


# -----------------------------------------
# SAVE DATA AND PREPROCESSING BUNDLE (TO DEDICATED FOLDER)
# -----------------------------------------
print(f"\nCreating save directory: '{SAVE_DIR}'")
os.makedirs(SAVE_DIR, exist_ok=True) # Create the 'data' folder

print("Saving processed data and preprocessing bundle...")

# Define full paths for saving
X_TRAIN_PATH = os.path.join(SAVE_DIR, "X_train.npy")
X_TEST_PATH = os.path.join(SAVE_DIR, "X_test.npy")
Y_TRAIN_PATH = os.path.join(SAVE_DIR, "y_train.npy")
Y_TEST_PATH = os.path.join(SAVE_DIR, "y_test.npy")
# --- CONSOLIDATED PICKLE PATH ---
LE_BUNDLE_PATH = os.path.join(SAVE_DIR, LE_FILENAME)


# Save feature data
np.save(X_TRAIN_PATH, X_train)
np.save(X_TEST_PATH, X_test)
np.save(Y_TRAIN_PATH, y_train)
np.save(Y_TEST_PATH, y_test)

# **CORRECT CONSOLIDATED PICKLE SAVE**
# Bundle both necessary objects into a dictionary
preprocessing_bundle = {
    'label_encoder': le,
    'feature_scaler': scaler
}
# Save the dictionary to the designated filename
joblib.dump(preprocessing_bundle, LE_BUNDLE_PATH) 


print(f"\n✅ All data saved safely to the '{SAVE_DIR}' folder!")
print(f"Files saved: X_*.npy, y_*.npy, and {LE_FILENAME} (containing scaler and label encoder)")