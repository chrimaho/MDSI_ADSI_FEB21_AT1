def print_reg_perf(y_preds, y_actuals, set_name=None):
    """
    Print the RMSE and MAE for the provided data.
    Source: Lecture notes.

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    
    # print(f"RMSE {set_name}: {mse(y_actuals, y_preds, squared=False)}")
    # print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")
    
    print("{name} RMSE: {score}".format(name=set_name, score=mse(y_actuals, y_preds, squared=False)))
    print("{name} MAE: {score}".format(name=set_name, score=mae(y_actuals, y_preds)))
    
def save_reg_perf(pred, targ, data_reg_metrics=None, overwrite=True):
    
    if data_reg_metrics==None:
        print("NaN")
    """
    Metrics:
    - MSE
    - RMSE
    - MAE
    - MAPE
    - R2
    - AR2
    """
    
    # Return
    return None