from xgboost import XGBClassifier

def get_xgboost_model(num_classes):
    """Defines and returns the XGBoost classifier."""
    model = XGBClassifier(
        objective='multi:softmax', 
        num_class=num_classes,
        use_label_encoder=False, 
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=42 # Added for reproducibility
    )
    return model