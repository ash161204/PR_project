Machine Learning Pipeline for Preprocessed Feature-Based Classification

📌 Overview

This project implements a complete machine-learning workflow for feature-based classification.
It includes:

A feature extraction + preprocessing pipeline

A training script that produces .pkl model files

A quick evaluation script that runs already-trained models

Organized folder structure for data, models, results, and source code

The project is designed so that full training (which takes several minutes) is only done when necessary, while main.py provides fast evaluation using pre-trained models.

📂 Project Structure
project/
│
├── data/                         # Preprocessed feature data (ready for training)
├── models/                       # Saved trained models (.pkl)
├── results/                      # Evaluation reports, plots, metrics
├── src/                          # Optional helper modules
│
├── feature_extraction_and_prep.py # Extract features & preprocess raw data
├── train.py                       # Train ML models and save .pkl files
├── main.py                        # Quick testing using pre-trained models
├── requirements.txt               # Python dependencies
└── .gitattributes

⚙️ Installation

Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

🛠️ Usage
1️⃣ Preprocess Data (Only When Raw Data Changes)

Run this if you added new raw data or want to regenerate processed features:

python feature_extraction_and_prep.py


This script saves the cleaned + feature-engineered dataset into the data/ directory.

2️⃣ Train Models

This step trains models like XGBoost, RandomForest, etc., using the preprocessed data.

python train.py


The training process may take several minutes.
All trained models are stored as .pkl files in the models/ folder.

3️⃣ Quick Evaluation (Recommended)

To quickly test performance without retraining:

python main.py


main.py loads the already-trained .pkl files and evaluates them on the test set.

No long training required — ideal for verification and debugging.

📊 Outputs
After training (train.py):

Saved models in models/

Optional training metrics in results/

After testing (main.py):

Accuracy, precision/recall/F1

Saved plots or confusion matrices inside results/

✔️ Notes

Ensure that train.py and main.py use the same feature ordering to avoid incorrect predictions.

Only run the feature extraction script when raw data changes.

For fast development cycles, prefer using main.py.
