import os
import numpy as np
import joblib
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings that often occur during model loading/evaluation
warnings.filterwarnings("ignore")

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------
DATA_DIR = "data"
MODELS_DIR = "models"
# --- NEW: Directory for saving visualization results ---
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True) 

# File paths
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.npy")
# --- PATH TO CONSOLIDATED FILE ---
PREPROCESSING_BUNDLE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")

# List of models to evaluate (matching the filenames saved in train.py)
MODEL_NAMES = ["LR", "SVM", "RF", "XGB"] 

# -----------------------------------------
# DATA LOADING
# -----------------------------------------
try:
    X_test  = np.load(X_TEST_PATH)
    y_test  = np.load(Y_TEST_PATH)
    
    # Load the consolidated bundle and extract the label encoder
    bundle = joblib.load(PREPROCESSING_BUNDLE_PATH)
    le = bundle['label_encoder'] 
    TARGET_NAMES = le.classes_
    
    print("✅ Test Data and Label Encoder loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ Error: Required data file not found in '{DATA_DIR}'. Please ensure 'feature_extraction_and_prep.py' was run.")
    print(f"Missing file: {e}")
    exit()

# -----------------------------------------
# PLOTTING FUNCTION
# -----------------------------------------

def plot_and_save_confusion_matrix(cm, name, target_names):
    """Generates and saves a heatmap visualization of the Confusion Matrix."""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=target_names, 
        yticklabels=target_names
    )
    plt.title(f'Confusion Matrix for {name} Model\nOverall Accuracy: {np.sum(np.diag(cm)) / np.sum(cm):.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Define save path
    save_path = os.path.join(RESULTS_DIR, f'{name}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close() # Close plot to free memory
    print(f"🖼️ Confusion Matrix saved to: {save_path}")


# -----------------------------------------
# MODEL EVALUATION EXECUTION
# -----------------------------------------

def evaluate_model(name, X_test, y_test, target_names):
    """Loads, evaluates, reports metrics, and saves CM for a single model."""
    model_filename = os.path.join(MODELS_DIR, f"{name}.pkl")
    
    print(f"\n=======================================================")
    print(f"   EVALUATING MODEL: {name}")
    print(f"=======================================================")
    
    if not os.path.exists(model_filename):
        print(f"❌ Model file not found: {model_filename}. Skipping evaluation.")
        return

    # Load Model
    try:
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading model {name}: {e}. Skipping evaluation.")
        return

    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # --- REPORTING (Terminal Output) ---
    print(f"Overall Accuracy: {acc:.4f}\n")
    
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # --- SAVING (File Output) ---
    plot_and_save_confusion_matrix(cm, name, target_names)


# Run the evaluation loop for all models
for name in MODEL_NAMES:
    evaluate_model(name, X_test, y_test, TARGET_NAMES)

print("\n✨ All model evaluations and result saving complete.")
