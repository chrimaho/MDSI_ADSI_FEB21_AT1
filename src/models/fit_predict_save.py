# Define function to fit and save models, and run predictions on validation data
def fit_predict_save(reg, name, X_train, y_train, X_val, y_val):
    
    """
    Args:
        reg: Classifier with parameters defined.
        name: Name given to model (to be saved to .joblib file)
        X_train: Numpy array of input values in training data
        y_train: Numpy array of actual target values in training data
        X_val: Numpy array of input values in validation data
        y_val: Numpy array of actual target values in validation data
    """
    
    # Fit classifier
    reg.fit(X_train, y_train)
    
    global y_train_pred, y_val_pred, y_val_prob
    
    # Model predictions on training and validation data
    y_train_pred = reg.predict(X_train)
    y_val_pred = reg.predict(X_val)
    y_val_prob = reg.predict_proba(X_val)[:,1]
    
    # Save fitted model into model folder
    save_path = ('../../models/'"%s"'.joblib' % name) 
    dump(reg, save_path)
    
    # Calculate AUC
    roc_auc_val = roc_auc_score(y_val, y_val_prob)
    print("\nROC_AUC:", roc_auc_val, "\n")
    
    return None