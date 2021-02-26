import numpy as np
import pandas as pd
from datetime import datetime

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
    
def get_auc(targ:np.real, pred_prob:np.real):
    """
    Get the ROC AUC score from a given probability distribution.
    Args:
        targ (np.real): The true classes.
        pred_prob (np.real): The probability of the true classes. Calculated from using the `Estimator.predict_proba(Y_val)` function.
        
    Raises:
        Assertions: Each parameter will be asserted to the correct type.
    Returns:
        float: The calculated AUC score.
    """

    # Imports
    import numpy as np
    from sklearn.metrics import roc_curve, auc

    # Assertions
    assert np.all(np.isreal(targ))
    assert np.all(np.isreal(pred_prob))

    # Correct
    if len(pred_prob.shape)>1:
        pred_prob = pred_prob[:,1]

    # Compute
    fpr, tpr, thresholds = roc_curve(targ, pred_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Return
    return roc_auc

def print_conf_matrix(targ:np.real, pred:np.real):
    """
    Print the Confusion Matrix for a given dataset
    Args:
        targ (np.real): The target values.
        pred (np.real): The predicted values.
    Raises:
        Assertions: The input parameters are asserted to the correct type and attribute.
    Returns:
        None: Nothing is returned
    """
    
    # Imports
    import sys
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix as conf
    from IPython.display import display
    
    # Assertions
    assert np.all(np.isreal(targ))
    assert np.all(np.isreal(pred))
    
    # Make data
    data = pd.DataFrame \
        ( conf(targ,pred)
        , columns=pd.MultiIndex.from_tuples([("pred",0),("pred",1)])
        , index=pd.MultiIndex.from_tuples([("targ",0),("targ",1)])
        )
        
    # display() in Jupyter, print() otherwise.
    if hasattr(sys, "ps1"):
        display(data)
    else:
        print(data)
        
    # Nothing to return
    return None

def plot_roc_curve(targ:np.real, pred_prob:np.real):
    """
    Plot the ROC curve from a given probability distribution.
    Args:
        targ (np.real): The true scores.
        pred_prob (np.real): The probability of the true scores. Calculated from using the `Estimator.predict_proba(Y_val)` function.
    Returns:
        None: Nothing is returned from this function because the plot is printed.
    """

    # Imports
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    from matplotlib import pyplot as plt

    # Assertions
    assert np.all(np.isreal(targ))
    assert np.all(np.isreal(pred_prob))

    # Generate data
    fpr, tpr, thresholds = roc_curve(targ, pred_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Generate plot
    plt.figure()
    lw=2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC (AUC)={:.3f}".format(roc_auc))
    plt.plot([0,1], [0,1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristics")
    plt.legend(loc="lower right")
    plt.show

    # Return
    return None


def save_reg_perf \
    ( targ:np.real
    , pred:np.real
    , pred_prob:np.real
    , df_metrics:pd.DataFrame
    , name:str=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    , overwrite:bool=True
    , print_all:bool=True
    , print_matrix:bool=True
    , print_plot:bool=True
    , print_df:bool=True
    ):
    """
    Save model metrics to a dataframe.
    Args:
        targ (np.real)                : The actual values. Can be scalar or arrya, but must be Real numbers.
        pred (np.real)                : The prediction values. Can be scalar or array, but must be Real numbers. Must provide _either_ `pred` or `pred_probs`. Defaults to None.
        pred_prob (np.real)           : The prediction probability values. Can be scalar or array, but must be Real numbers. Must provide _either_ `pred` or `pred_probs`. Defaults to None.
        df_metrics (pd.DataFrame)     : The data frame to be updated to contain the model metrics.
        name (str, optional)          : The name of the data being calculated. If not given, will default to 'None'. Defaults to datetime.now().strftime("%Y-%m-%d %H:%M:%S").
        overwrite (bool, optional)    : Whether or not to overwrite the data in the dataframe. In SQL-speak: True=UPDATE, False=INSERT. Defaults to True.
        print_all (bool, optional)    : Wheather or not to turn off all printing. Defaults to True.
        print_matrix (bool, optional) : Whether or not to print the confusion matrix. Defaults to True.
        print_plot (bool, optional)   : Whether or not to print the ROC plot. Defaults to True.
        print_df (bool, optional)     : Whether or not to print the final updated dataframe. Defaults to True.
    Raises:
        Assertions: Each parameter will be asserted to the proper type and attribute.
    Returns:
        pd.DataFrame: The updated dataframe.
    """    
    
    # Imports
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix as conf
    from sklearn.metrics import roc_auc_score as roc_auc
    from sklearn.metrics import f1_score as f1
    from src.utils.misc import all_in
    from IPython.display import display
    from src.models.performance import print_conf_matrix, plot_roc_curve

    # Assertions
    assert np.all(np.isreal(targ))
    assert np.all(np.isreal(pred))
    assert np.all(np.isreal(pred_prob))
    assert np.isscalar(name)
    assert isinstance(name, str)
    assert isinstance(df_metrics, pd.DataFrame)
    assert all_in(df_metrics.columns, ["name","when","auc","sens","spec","f1"])
    for parameter in [overwrite, print_matrix, print_plot, print_df]:
        assert isinstance(parameter, bool)

    # Start your engines
    df = df_metrics

    # Fix dimensions of the prob part
    if len(pred_prob.shape)>1:
        pred_prob = pred_prob[:,1]

    # Perform calculations
    val_now = pd.Timestamp.now().strftime('%d/%b %H:%M')
    val_auc = round(roc_auc(targ,pred_prob), 5)
    # vau_auc = round(get_auc(targ,pred_prob), 5)
    tn, fp, fn, tp = conf(targ,pred).ravel()
    val_sens = round(tp/(tp+fn), 5)
    val_spec = round(tn/(tn+fp), 5)
    val_f1 = round(f1(targ,pred), 5)

    # Two different methods of updating the table. In SQL-Speak this is the difference between INSERT and UPDATE
    if overwrite and name in df["name"].to_numpy():
        df.loc[df["name"] == name, ["when"]] = val_now
        df.loc[df["name"] == name, ["auc"]] = val_auc
        df.loc[df["name"] == name, ["sens"]] = val_sens
        df.loc[df["name"] == name, ["spec"]] = val_spec
        df.loc[df["name"] == name, ["f1"]] = val_f1
    else:
        new = pd.DataFrame \
            (
                { "name": [name]
                , "when": [val_now]
                , "auc": [val_auc]
                , "sens": [val_sens]
                , "spec": [val_spec]
                , "f1": [val_f1]
                }
            )
        df = df.append(new)

    # Fix Pandas indexes
    df.reset_index(drop=True, inplace=True)

    # Print if needed
    if print_all:
        if print_matrix:
            print_conf_matrix(targ, pred)
        if print_plot:
            plot_roc_curve(targ, pred_prob)
        if print_df:
            display(df)

    # Return
    return df
