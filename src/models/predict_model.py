import numpy as np
import pandas as pd
from datetime import datetime

# Define reusable function for easy random searching
def easy_random_search \
    ( estimator
    , search_space:dict
    , feat_trn:np.real
    , targ_trn:np.real
    , feat_val:np.real
    , targ_val:np.real
    , df_metrics:pd.DataFrame
    , n_iter:int=100
    , cv:int=5
    , random_state:int=123
    , check_best_params:bool=True
    , dump_model:bool=True
    , dump_location:str="./models/Chris/"
    , dump_name:str=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    , print_all:bool=True
    , print_matrix:bool=True
    , print_plot:bool=True
    , print_df:bool=True
    ):
    """
    Quickly and easily re-run the Random Search algorithm to find the optimal parameters and see the model results.
    Args:
        estimator (estimator): An estimator to be used for training. Must be instantiated!
        search_space (dict)                : The search space to be checked. The keys must be valid hyperparameters in the `estimator` object.
        feat_trn (np.real)                 : The features to be used for training.
        targ_trn (np.real)                 : The target values to be used for training.
        feat_val (np.real)                 : The features to be used for validation.
        targ_val (np.real)                 : The target values to be used for validation.
        df_metrics (pd.DataFrame)          : The data frame to be updated to contain the model metrics.
        n_iter (int, optional)             : Number of times the Search Space is to be checked. Defaults to 100.
        cv (int, optional)                 : Number of cross-validations to be run per iteration. Defaults to 5.
        random_state (int, optional)       : The random state to be used for the `cv` splitting. Defaults to 123.
        check_best_params (bool, optional) : Whether or not to print the best params from the search space after training. Defaults to True.
        dump_model (bool, optional)        : Whether or not to dump the model after training. Defaults to True.
        dump_location (str, optional)      : The location where the model should be dumped to. Defaults to "./models/Chris/".
        dump_name (str, optional)          : The file name of the model once dumped. Defaults to datetime.now().strftime("%Y-%m-%d %H:%M:%S").
        print_all (bool, optional)         : Whether or not to print all the results & metrics. Defaults to True.
        print_matrix (bool, optional)      : Whether or not to print the confusion matrix. Defaults to True.
        print_plot (bool, optional)        : Whether or not to print the ROC plot. Defaults to True.
        print_df (bool, optional)          : Whether or not to print the dataframe with the results from all models for all metrics. Defaults to True.
    Raises:
        Assertions: All parameters are asserted to the correct type and correct attributes.
    
    Returns:
        estimator: The re-trained model, using the best params from the search space.
    """

    # Imports
    from sklearn.model_selection import RandomizedSearchCV
    from src.utils.misc import all_in
    from src.utils.performance import TicToc
    from src.models.performance import save_reg_perf
    import numpy as np
    from xgboost.sklearn import XGBModel
    from sklearn.metrics import make_scorer, roc_auc_score
    import os
    from joblib import dump

    # Instantiate timer
    t = TicToc()

    # Assertions
    # assert "base_estimator" in estimator.__dict__.keys()
    # assert "sklearn" in estimator.__module__.split(".")[0]
    assert isinstance(search_space, dict), \
        "`search_space` must be type `dict`."
    assert all_in(search_space.keys(), estimator.__dict__.keys()), \
        "All keys in `search_space` must be valid parameters in `estimator`."
    for param in ["feat_trn", "targ_trn", "feat_val", "targ_val"]:
        assert isinstance(eval(param), np.ndarray), \
            "`{param}` must be type `np.ndarray`."
        assert np.all(np.isreal(eval(param))), \
            "All elements of `{param}` must be Real numbers."
    assert len(feat_trn)==len(targ_trn), \
        "Lengh of `feat_trn` must be same as `targ_trn`."
    assert len(feat_val)==len(targ_val), \
        "Length of `feat_val` must be same as `targ_val`."
    for param in ["n_iter", "cv", "random_state"]:
        assert isinstance(eval(param), int), \
            "`{param}` must be type `int`."
        assert eval(param)>0, \
            "`{param}` must be a positive integer."
    for param in ["check_best_params", "dump_model", "print_all", "print_matrix", "print_plot", "print_df"]:
        assert isinstance(eval(param), bool), \
            "`{param}` must be type `bool`."
    for param in ["dump_location", "dump_name"]:
        assert isinstance(eval(param), str), \
            "`{param}` must be type `str`."
    assert os.path.isdir(dump_location), \
        "`dump_location` must be a valid direcory."

    # Instantiate trainer
    clf = RandomizedSearchCV \
        ( estimator=estimator
        , param_distributions=search_space
        , n_iter=n_iter
        , scoring={"auc": make_scorer(roc_auc_score, needs_proba=True)}
        , cv=cv
        , refit="auc"
        , random_state=random_state
        , return_train_score=True
        )
    
    # Search for results
    t.tic()
    if isinstance(estimator, XGBModel):
        res = clf.fit(feat_trn, targ_trn, eval_metric="auc")
    else:
        res = clf.fit(feat_trn, targ_trn)
    t.toc()

    # Check best params
    if check_best_params:
        print("Best score: {}".format(res.best_score_))
        print("Best params: {}".format(res.best_params_))
    
    # Update params
    estimator = estimator.set_params(**res.best_params_)

    # Refit
    if isinstance(estimator, XGBModel):
        estimator.fit(feat_trn, targ_trn, eval_metric="auc")
    else:
        estimator.fit(feat_trn, targ_trn)

    # Predict
    pred_trn = estimator.predict(feat_trn)
    pred_prob_trn = estimator.predict_proba(feat_trn)
    pred_val = estimator.predict(feat_val)
    pred_prob_val = estimator.predict_proba(feat_val)

    # Check performance
    df_metrics = save_reg_perf \
        ( targ=targ_trn
        , pred=pred_trn
        , pred_prob=pred_prob_trn
        , df_metrics=df_metrics
        , name=dump_name+" - within bag"
        , print_all=False
        , print_matrix=print_matrix
        , print_plot=print_plot
        , print_df=print_df
        )
    df_metrics = save_reg_perf \
        ( targ=targ_val
        , pred=pred_val
        , pred_prob=pred_prob_val
        , df_metrics=df_metrics
        , name=dump_name+" - out of bag"
        , print_all=print_all
        , print_matrix=print_matrix
        , print_plot=print_plot
        , print_df=print_df
        )

    # Backup
    if dump_model:
        dump(estimator, dump_location+dump_name+".joblib")

    # Return
    return estimator, df_metrics
