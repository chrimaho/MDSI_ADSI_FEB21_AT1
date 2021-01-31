# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # The Future of Basketball Players
# %% [markdown]
# ## Set up Environment

# %%
# get_ipython().system('python -c "import pandas as pd; print(pd.__version__)"')


# %%
# get_ipython().system('pip show scipy')


# %%
import os
import sys
import zipfile
import sklearn
import pandas as pd
import numpy as np
from joblib import dump
from joblib import load
from pprint import pprint
from pandas_profiling import ProfileReport
from IPython.display import IFrame

if not "Interactive" in get_ipython().__class__.__name__:
    os.chdir("./notebooks/Chris")

# Add current project to Python Sys Path
if not os.path.abspath("../..") in sys.path:
    sys.path.append(os.path.abspath("../.."))
else:
    sys.path.remove(os.path.abspath("../.."))
    sys.path.append(os.path.abspath("../.."))

# %% [markdown]
# ### Custom Global Functions

# %%
def str_right(string:str, num_chars:int):
    """ Sub-Select the right-most number of characters from a string """
    # Assertions
    assert isinstance(string, str)
    assert isinstance(num_chars, int)
    # Return
    return string[-num_chars:]

# %% [markdown]
# ## Get Data

# %%

# Check & Create directories
for dir in ["../../data/processed/Chris", "../../models/Chris"]:
    if not os.path.exists(dir): os.mkdir(dir)


# %%
# Use bash to get data from kaggle
# get_ipython().system('kaggle competitions files -c uts-advdsi-nba-career-prediction')
# get_ipython().system('kaggle competitions download -c uts-advdsi-nba-career-prediction --path ../../data/raw --force --quiet')

# Unzip the data
# with zipfile.ZipFile("../../data/raw/uts-advdsi-nba-career-prediction.zip", "r") as z:
#     z.extractall("../../data/raw")


# Delete the zip file
if os.path.exists("../../data/raw/uts-advdsi-nba-career-prediction.zip"):
    os.remove("../../data/raw/uts-advdsi-nba-career-prediction.zip")


# %%
# Turn off table squishing
pd.options.display.max_columns = None

# Import data
data = pd.read_csv("../../data/raw/train.csv")

# Check data
# print(data.columns)
# # displaydata)


# %%
# Fix column names
print(data.columns)
data.columns = data.columns.str.replace(' ', '_')
print(data.columns)

# Save training data again
data.to_csv("../../data/raw/train.csv", index=False)

# Apply same logix to test data
test = pd.read_csv("../../data/raw/test.csv")
test.columns = test.columns.str.replace(' ', '_')
test.to_csv("../../data/raw/test.csv", index=False)

# %% [markdown]
# ## Check Data
# %% [markdown]
# ### Data Dictionary

# %%
# Direct copy & paste from:
# https://www.kaggle.com/c/uts-advdsi-nba-career-prediction/data

DataDict =     { "Id": "Player Identifier"
    , "GP": "Games Played"
    , "MIN": "Minutes Played"
    , "PTS": "Points Per Game"
    , "FGM": "Field Goals Made"
    , "FGA": "Field Goals Attempts"
    , "FG": "Field Goals Percent"
    , "3P_Made": "3-Points Made"
    , "3PA": "3-Points Attempts"
    , "3P": "3-Points Percent"
    , "FTM": "Free Throw Made"
    , "FTA": "Free Throw Attempts"
    , "FT": "Free Throw Percent"
    , "OREB": "Offensive Rebounds"
    , "DREB": "Defensive Rebounds"
    , "REB": "Rebounds"
    , "AST": "Assists"
    , "STL": "Steals"
    , "BLK": "Blocks"
    , "TOV": "Turnovers"
    , "TARGET_5Yrs": "Outcome: 1 if career length >= 5 years, 0 otherwise"
    }

# %% [markdown]
# ### Data Exploration

# %%
# Prevent the need to re-run
if os.path.exists("InitialReport.html"):
    IFrame(src="InitialReport.html", width="100%", height=600)
else:
    # Create profile report
    profile = ProfileReport(data, title="Profile Report")
    # Export
    profile.to_file("InitialReport.html")
    # View
    profile

# %% [markdown]
# ## Experimentation
# %% [markdown]
# ### Data Manipulation
# 
# Steps:
# 
# 1. Drop the `ID` column
# 1. Pop the `target` column
# 1. Scale the features
# 1. Split in to train/test

# %%
# Drop ID
data = data.drop(['Id'], axis=1)

# Pop targ
feat = data.iloc[:,:-1].to_numpy()
targ = data.iloc[:,-1].to_numpy()

# Scale feat
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
feat = scaler.fit_transform(feat)

# Tran/Test split
from sklearn.model_selection import train_test_split
feat_trn, feat_val, targ_trn, targ_val = train_test_split(feat, targ, test_size=0.2, random_state=6)


# %%
# Dump your objects
dump(data, "../../data/processed/Chris/data.joblib")
dump(feat, "../../data/processed/Chris/feat.joblib")
dump(targ, "../../data/processed/Chris/targ.joblib")
dump(feat_trn, "../../data/processed/Chris/feat_trn.joblib")
dump(feat_val, "../../data/processed/Chris/feat_val.joblib")
dump(targ_trn, "../../data/processed/Chris/targ_trn.joblib")
dump(targ_val, "../../data/processed/Chris/targ_val.joblib")
dump(scaler, "../../models/Chris/scaler.joblib")

# %% [markdown]
# ### Set up Experiment Space

# %%
# Set DF for score dumps
pred_scor = pd.DataFrame(
    { "name": []
    , "when": []
    , "mse": []
    , "rmse": []
    , "mae": []
    , "mape": []
    , "r2": []
    }
)


# %%
def save_reg_perf(pred:np.real, targ:np.real, name:str=None, df_metrics:pd.DataFrame=None, overwrite:bool=True, print:bool=True):
    """
    Save model metrics to a dataframe.

    Args:
        pred (np.real): The prediction values. Can be scalar or array, but must be Real numbers.
        targ (np.real): The actual values. Can be scalar or arrya, but must be Real numbers.
        name (str, optional): The name of the data being calculated. If not given, will default to 'None'. Defaults to None.
        df_metrics (pd.DataFrame, optional): The data frame to be updated to contain the data. If not given, will update the `pred_scor` dataframe from the global scope. Defaults to None.
        overwrite (bool, optional): Whether or not to overwrite the data in the dataframe. In SQL-speak: True=UPDATE, False=INSERT. Defaults to True.
        print (bool, optional): Whether or not to print the final updated dataframe. Defaults to True.

    Raises:
        Assertions: Each parameter will be asserted to the proper type and attribute.

    Returns:
        pd.DataFrame: The updated dataframe.
    """    
    
    # Perform assertions
    # assert np.isscalar(pred)
    assert np.all(np.isreal(pred))
    # assert np.isscalar(targ)
    assert np.all(np.isreal(targ))
    assert isinstance(overwrite, bool)
    assert isinstance(print, bool)
    if not name==None:
        assert np.isscalar(name)
        assert isinstance(name, str)
    if not df_metrics==None:
        assert isinstance(df_metrics, pd.DataFrame)
        assert df_metrics.columns == ["name","when","mse","rmse","mae","mape","r2"]
    
    # Import modules
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import mean_absolute_percentage_error as mape
    from sklearn.metrics import r2_score as r2

    # Ensure we're using the global object here
    global pred_scor

    # If you want to use another dataframe, go right ahead. Else, just keep it simple.
    if df_metrics==None: df = pred_scor
    else: df = df_metrics
    
    # Best to define name, but if blank then make None
    if name==None: name=="None"

    # Perform calculations
    val_now = pd.Timestamp.now().strftime('%d/%b %H:%M')
    val_mse = round(mse(targ, pred), 5)
    val_rmse = round(mse(targ, pred, squared=False), 5)
    val_mae = round(mae(targ, pred), 5)
    val_mape = round(mape(targ, pred), 5)
    val_r2 = round(r2(targ, pred), 5)

    # Two different methods of updating the table. In SQL-Speak this is the difference between INSERT and UPDATE
    if overwrite and name in df["name"].to_numpy():
        df.loc[df["name"] == name, ["when"]] = val_now
        df.loc[df["name"] == name, ["mse"]] = val_mse
        df.loc[df["name"] == name, ["rmse"]] = val_rmse
        df.loc[df["name"] == name, ["mae"]] = val_mae
        df.loc[df["name"] == name, ["mape"]] = round(val_mape, 5)
        df.loc[df["name"] == name, ["r2"]] = val_r2
    else:
        new = pd.DataFrame(
            { "name": [name]
            , "when": [val_now]
            , "mse": [val_mse]
            , "rmse": [val_rmse]
            , "mae": [val_mae]
            , "mape": [val_mape]
            , "r2": [val_r2]
            }
        )
        df = df.append(new)

    # Fix Pandas indexes
    df.reset_index()

    # Assign back to the global scope
    pred_scor = df

    # Print if needed
    # if print:
        # displaydf)

    # Return
    return df

# %% [markdown]
# ### Baseline

# %%
# Add current project to Python Sys Path
if not os.path.abspath("../..") in sys.path:
    sys.path.append(os.path.abspath("../.."))
else:
    sys.path.remove(os.path.abspath("../.."))
    sys.path.append(os.path.abspath("../.."))

# Get mean of target
avg = data['TARGET_5Yrs'].mean()

# Generate actuals and baseline
bas = np.resize(avg, data.shape[0])
act = np.array(data['TARGET_5Yrs'])

# Check objects
# from src.utils.objects import check_object
# check_object(act, "act")
# check_object(bas, "bas")

# Check performance
save_reg_perf(bas, act, "Raw baseline", print=False)
# # displaypred_scor)

# %% [markdown]
# ### Attempt #1: ElasticNet

# %%
# Import model
from sklearn.linear_model import ElasticNet

# Instantiate
reg = ElasticNet()

# Train
reg.fit(feat_trn, targ_trn)

# Predict on training data
pred_trn = reg.predict(feat_trn)

# Predict on validation data
pred_val = reg.predict(feat_val)

# Check performance
save_reg_perf(pred_trn, targ_trn, "ElasticNet - Defaults - within bag", print=False)
save_reg_perf(pred_val, targ_val, "ElasticNet - Defaults - out of bag", print=False)
# displaypred_scor)

# Backup
dump(reg, "../../models/Chris/01_ElasticNet_Defaults.joblib")


# %%
# Marginally better... barely worth it...

# %% [markdown]
# ### Attempt #2: ElasticNetCV

# %%
# Import model
from sklearn.linear_model import ElasticNetCV

# Instantiate
reg = ElasticNetCV()

# Train
reg.fit(feat, targ)

# Predict on validation data
pred = reg.predict(feat)

# Check performance
save_reg_perf(pred, targ, "ElasticNetCV - Defaults")

# Backup
dump(reg, "../../models/Chris/02_ElasticNetCV_Defaults.joblib")


# %%
# Slignt improvement. But nothing substantive yet.

# %% [markdown]
# ### Attempt #3: Random Forest

# %%
# Import model
from sklearn.ensemble import RandomForestRegressor

# Instantiate
reg = RandomForestRegressor(random_state=123)

# Train
reg.fit(feat_trn, targ_trn)

# Predict on training data
pred_trn = reg.predict(feat_trn)

# Predict on validation data
pred_val = reg.predict(feat_val)

# Check performance
save_reg_perf(pred_trn, targ_trn, "RandomForest - Defaults - within bag", print=False)
save_reg_perf(pred_val, targ_val, "RandomForest - Defaults - out of bag", print=False)
# displaypred_scor)

# Backup
dump(reg, "../../models/Chris/03_RandomForest_Defaults.joblib")


# %%
# **NOW** We're cookin with Gas!

# %% [markdown]
# ### Attempt #4: Bagging Algo

# %%
# Import model
from sklearn.ensemble import BaggingRegressor

# Instantiate
reg = BaggingRegressor(random_state=123)

# Train
reg.fit(feat_trn, targ_trn)

# Predict on training data
pred_trn = reg.predict(feat_trn)

# Predict on validation data
pred_val = reg.predict(feat_val)

# Check performance
save_reg_perf(pred_trn, targ_trn, "BaggingAlgo - Defaults - within bag", print=False)
save_reg_perf(pred_val, targ_val, "BaggingAlgo - Defaults - out of bag", print=False)
# displaypred_scor)

# Backup
dump(reg, "../../models/Chris/04_BaggingAlgo_Defaults.joblib")


# %%


# %% [markdown]
# ### Attempt #5: Gradient Boosting

# %%
# Import model
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate
reg = GradientBoostingRegressor(random_state=123)

# Train
reg.fit(feat_trn, targ_trn)

# Predict on training data
pred_trn = reg.predict(feat_trn)

# Predict on validation data
pred_val = reg.predict(feat_val)

# Check performance
save_reg_perf(pred_trn, targ_trn, "GBM - Defaults - within bag", print=False)
save_reg_perf(pred_val, targ_val, "GBM - Defaults - out of bag", print=False)
# displaypred_scor)

# Backup
dump(reg, "../../models/Chris/05_GBM_Defaults.joblib")


# %%
# Slightly better than the ElasticNet models, not as good as the RandomForest

# %% [markdown]
# ### Attempt #6: Ada Boost

# %%
# Import model
from sklearn.ensemble import AdaBoostRegressor

# Instantiate
reg = AdaBoostRegressor(random_state=123)

# Train
reg.fit(feat_trn, targ_trn)

# Predict on training data
pred_trn = reg.predict(feat_trn)

# Predict on validation data
pred_val = reg.predict(feat_val)

# Check performance
save_reg_perf(pred_trn, targ_trn, "AdaBoost - Defaults - within bag", print=False)
save_reg_perf(pred_val, targ_val, "AdaBoost - Defaults - out of bag", print=False)
# displaypred_scor)

# Backup
dump(reg, "../../models/Chris/06_AdaBoost_Defaults.joblib")


# %%



# %%



# %%



# %%



# %%


# %% [markdown]
# ## Final Processing & Submitting
# %% [markdown]
# ### Define final processing steps

# %%
def final_prediction_processing     ( final_model_path:str
    , data_scaler_path:str
    , test_data_path:str="../../data/raw/test.csv"
    , submission_data_path:str="../../data/external/final.csv"
    , check:bool=False
    ):
    """
    Process the data for the final prediction.
    Note, this will update the prediction and save it to the `submission_data_path` location.

    Args:
        final_model_path (str): The location from which the final model should be drawn from. Note, this model should be saved as a `` file type, and once imported, should be a sklearn model class.
        data_scaler_path (str): The location from which the data scaler should be imported from. Note, this should be a sklearn preprocessing class
        test_data_path (str, optional): The location from which the test data should be imported. Note, it should be a `.csv` file type. Defaults to "../../data/raw/test.csv".
        submission_data_path (str, optional): The location to which the final submission data should be stored. Note, it should be a `.csv` file type. Defaults to "../../data/external/final.csv".
        check (bool, optional): Optional flag for wheather or not to `# display)` the final data frame, mainly for checking purposes. Defaults to False.

    Returns:
        pd.DataFrame: The updated `submission_data` dataframe
    """

    # Imports
    import numpy as np
    import pandas as pd

    # Assertions
    for parameter in [final_model_path, data_scaler_path, test_data_path, submission_data_path]:
        assert np.isscalar(parameter)
        assert isinstance(parameter, str)
        if parameter in [final_model_path, data_scaler_path]:
            assert str_right(parameter, 7) == ".joblib"
        if parameter in [test_data_path, submission_data_path]:
            assert str_right(parameter, 4) == ".csv"
    assert isinstance(check, bool)
    
    # Import selected data
    final_model = load(final_model_path)
    data_scaler = load(data_scaler_path)
    test_data = pd.read_csv(test_data_path)

    # Process final data
    submission_data = test_data.pop("Id")
    test_data = data_scaler.fit_transform(test_data)

    # Get prediction
    final_prediction = final_model.predict(test_data)

    # Form final data
    submission_data = pd.concat([pd.DataFrame(submission_data), pd.DataFrame(final_prediction)], axis=1)
    submission_data.columns = ["Id", "TARGET_5Yrs"]
    submission_data["TARGET_5Yrs"] = submission_data["TARGET_5Yrs"].apply(lambda pred: 1 if pred > 0.5 else 0)

    # Check
    # if check:
        # display(submission_data)

    # Save to file
    submission_data.to_csv(submission_data_path, index=False)

    return submission_data


# %%
def final_prediction_processing     ( final_model_path:str
    , data_scaler_path:str
    , test_data_path:str="../../data/raw/test.csv"
    , submission_data_path:str="../../data/external/final.csv"
    , check:bool=False
    ):
    """
    Process the data for the final prediction.
    Note, this will update the prediction and save it to the `submission_data_path` location.

    Args:
        final_model_path (str): The location from which the final model should be drawn from. Note, this model should be saved as a `` file type, and once imported, should be a sklearn model class.
        data_scaler_path (str): The location from which the data scaler should be imported from. Note, this should be a sklearn preprocessing class
        test_data_path (str, optional): The location from which the test data should be imported. Note, it should be a `.csv` file type. Defaults to "../../data/raw/test.csv".
        submission_data_path (str, optional): The location to which the final submission data should be stored. Note, it should be a `.csv` file type. Defaults to "../../data/external/final.csv".
        check (bool, optional): Optional flag for wheather or not to `# display)` the final data frame, mainly for checking purposes. Defaults to False.

    Returns:
        pd.DataFrame: The updated `submission_data` dataframe
    """

    # Imports
    import numpy as np
    import pandas as pd

    # Assertions
    for parameter in [final_model_path, data_scaler_path, test_data_path, submission_data_path]:
        assert np.isscalar(parameter)
        assert isinstance(parameter, str)
        if parameter in [final_model_path, data_scaler_path]:
            assert str_right(parameter, 7) == ".joblib"
        if parameter in [test_data_path, submission_data_path]:
            assert str_right(parameter, 4) == ".csv"
    assert isinstance(check, bool)
    
    # Import selected data
    final_model = load(final_model_path)
    data_scaler = load(data_scaler_path)
    test_data = pd.read_csv(test_data_path)

    # Process final data
    submission_data = test_data.pop("Id")
    test_data = data_scaler.fit_transform(test_data)

    # Get prediction
    final_prediction = final_model.predict(test_data)

    # Form final data
    submission_data = pd.concat([pd.DataFrame(submission_data), pd.DataFrame(final_prediction)], axis=1)
    submission_data.columns = ["Id", "TARGET_5Yrs"]
    submission_data["TARGET_5Yrs"] = submission_data["TARGET_5Yrs"].apply(lambda pred: 1 if pred > 0.5 else 0)

    # Check
    # if check:
        # displaysubmission_data)

    # Save to file
    submission_data.to_csv(submission_data_path, index=False)

    return submission_data

# Process the final data
submission_data = final_prediction_processing     ( final_model_path = "../../models/Chris/03_RandomForest_Defaults.joblib"
    , data_scaler_path = "../../models/Chris/scaler.joblib"
    , test_data_path = "../../data/raw/test.csv"
    , submission_data_path = "../../data/external/final.csv"
    )

# %% [markdown]
# ### Submit to Kaggle

# %%
# Function to call Kaggle API
def call_kaggle_api(upload_file, upload_message):
    """
    Call the Kaggle API to submit a file
    Inspiration: https://unix.stackexchange.com/questions/190495/how-to-execute-a-bash-command-in-a-python-script#answer-579917
    
    Args:
        upload_file (str): The source path where the file will be uploaded _from_.
        upload_message (str): The message that will be sent to Kaggle for this submission.

    Raises:
        Exception: CallProcessError, incase subprocess has a wobbly
        Exception: Exception, for everything else

    Returns:
        str: The confirmation string from the API call
    """

    import subprocess

    bash_call = "kaggle competitions submit --competition uts-advdsi-nba-career-prediction --file {upload_file} --message {upload_message} --quiet".format(upload_file=upload_file, upload_message=upload_message)

    try:        
        result = subprocess.run(bash_call, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        raise Exception(str(err.stderr.decode("utf-8")))
    except Exception as err:
        raise Exception(err)
    else:
        return result.stdout.decode("utf-8")


# %%
# Call the API
if False:
    call_kaggle_api         ( upload_file = "../../data/external/final.csv"
        , upload_message = "Ctrl-Alt-Elite_RandomForest_Defaults"
        )


