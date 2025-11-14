from sklearn.ensemble import RandomForestClassifier

def get_random_forest_model():
    """Defines and returns the Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=40,
        max_features='log2',
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1,
        random_state=42 # Added for reproducibility
    )
    return model