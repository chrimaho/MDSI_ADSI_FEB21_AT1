# Define function to save model performance to a dataframe
def save_perf \
    (  name
     , y_val:np.real
     , y_val_pred:np.real
     , y_val_prob:np.real
     , overwrite:bool=True
     , print_dataframe: bool=True
     , save_dataframe:bool=True
    ):
    
    """
    Args:
        name: Name of model (saved to output table)
        y_val: Numpy array of actual target values in validation data.
        y_val_pred: Numpy array of target values predicted by model.
        y_val_prob: Numpy array of probability estimates for target values predicted by model.
        overwrite (bool, optional): Whether or not to overwrite the data in the dataframe.
        print_dataframe (bool, optional): Whether or not to print the final updated dataframe.
        save_dataframe (bool, optional): Whether or not to save the updated dataframe to disk as .csv file.
    """

    # Import modules
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    from sklearn.metrics import f1_score as f1
    
    # Define and calculate metrics
    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    val_auc = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    val_sens = round(tp/(tp+fn), 5)
    val_spec = round(tn/(tn+fp), 5)
    val_f1 = round(f1(y_val,y_val_pred), 5)
    
    global pred_scor
    
    df = pred_scor

    # Two different methods of updating the table
    if overwrite and name in df["name"].to_numpy():
        df.loc[df["name"] == name, ["auc"]] = val_auc
        df.loc[df["name"] == name, ["sens"]] = val_sens
        df.loc[df["name"] == name, ["spec"]] = val_spec
        df.loc[df["name"] == name, ["f1"]] = val_f1
    else:
        new = pd.DataFrame \
            (
                { "name": [name]
                , "auc": [val_auc]
                , "sens": [val_sens]
                , "spec": [val_spec]
                , "f1": [val_f1]
                }
            )
        
        df = pred_scor.append(new)
       
    # Fix Pandas indexes
    df.sort_values(by=['auc'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    

    # Assign back to the global scope
    pred_scor = df
    
    if print_dataframe:
        display(df)
        
    if save_dataframe:
        df.to_csv("../../models/experiment_results.csv")
    
    return None