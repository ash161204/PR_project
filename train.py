import os
import numpy as np
import joblib
import pickle
import time
from sklearn.metrics import accuracy_score

# Import model definitions from the src/models package
from src.logistic_regression import get_logistic_regression_model
from src.svm import get_svm_model
from src.random_forest import get_random_forest_model
from src.xgboost_clf import get_xgboost_model


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------
DATA_DIR = "data"
MODELS_DIR = "models"
# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True) 

# File paths
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.npy")
LABEL_ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")


# -----------------------------------------
# DATA LOADING
# -----------------------------------------
try:
    X_train = np.load(X_TRAIN_PATH)
    X_test  = np.load(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test  = np.load(Y_TEST_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    NUM_CLASSES = len(le.classes_)
    
    print(" Data loaded successfully from the 'data' folder!")

except FileNotFoundError as e:
    print(f" Error: Required data file not found. Have you run 'feature_extraction_and_prep.py' and ensured the data is in the '{DATA_DIR}' folder?")
    print(f"Missing file: {e}")
    exit()

# -----------------------------------------
# MODEL TRAINING EXECUTION
# -----------------------------------------

def train_and_save_model(name, model, X_train, y_train):
    """Generic function to train and save a sklearn model."""
    print(f"\n--- Training {name} ---")
    start_time = time.time()

    # Train
    model.fit(X_train, y_train)
    
    # Save
    model_filename = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    
    end_time = time.time()
    
    # Report only training completion time
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Model saved as {model_filename}")


# Dictionary of models to train
models_to_train = {
    "LR": get_logistic_regression_model(),
    "SVM": get_svm_model(),
    "RF": get_random_forest_model(),
    # XGBoost requires the number of classes
    "XGB": get_xgboost_model(NUM_CLASSES) 
}

# Run the training loop for all models
for name, model in models_to_train.items():
    train_and_save_model(name, model, X_train, y_train)

print("\n All model training complete! Check the 'models' folder for the saved pickle files.")
