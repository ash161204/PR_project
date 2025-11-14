from sklearn.linear_model import LogisticRegression

def get_logistic_regression_model():
    """Defines and returns the Logistic Regression classifier."""
    model = LogisticRegression(
        max_iter=10000,
        n_jobs=-1,
        multi_class='multinomial',
        random_state=42 # Added for reproducibility
    )
    return model