from sklearn.svm import LinearSVC

def get_svm_model():
    """Defines and returns the Linear Support Vector Machine (SVM) classifier."""
    model = LinearSVC(
        C=5.0, 
        penalty='l1', 
        dual=False, 
        max_iter=1000, # Increased max_iter for safety
        random_state=42 # Added for reproducibility
    )
    return model