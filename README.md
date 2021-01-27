# ADSI_AT1

MDSI - Advanced Data Science for Innovation - February 2021 - Assessment Task 01

## Kaggle

Site: [[UTS AdvDSI] NBA Career Prediction](https://www.kaggle.com/c/uts-advdsi-nba-career-prediction/overview)

## Command to set up repo

Main Command:

```
> cookiecutter --output-dir "C:\Users\chris\OneDrive\02 - Education\07 - MDSI\09 - ADSI\05 - Assessment\01\MDSI_ADSI_FEB21_AT1" https://github.com/drivendata/cookiecutter-data-science
```

Options:

```
> You've downloaded C:\Users\chris\.cookiecutters\cookiecutter-data-science before. Is it okay to delete and re-download it? [yes]: yes
> project_name [project_name]: ADSI_AT1
> repo_name [adsi_at1]: MDSI_ADSI_FEB21_AT1
> author_name [Your name (or your organization/company/team)]: Chris Mahoney
> description [A short description of teh project.]: MDSI - Advanced Data Science for Innovation - February 2021 - Assessment Task 01
> Select open_source_license:
1 - MIT
2 - BSD-3-Clause
3 - No license file
Choose from 1, 2, 3 [1]: 1
> s3_bucket [[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')]: 
> aws_profile [default]: 
> Select python_interpreter:
1 - python3
2 - python
Choose from 1, 2 [1]: 1
```

Then, once it loads, run:

```
> cd MDSI_ADSI_FEB21_AT1
> git init
> git add .
> git commit -m "Initial commit"
> git branch -M main
> git remote add origin https://github.com/chrimaho/MDSI_ADSI_FEB21_AT1.git
> git push -u origin main
```

(Check [here](https://docs.github.com/en/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line) or [here](https://stackoverflow.com/questions/54523848/github-setup-repository#answer-54524070) for details on git init process)

## Project Organization


```
.
└── MDSI_ADSI_FEB21_AT1
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
