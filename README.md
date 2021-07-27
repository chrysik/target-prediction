### RHI Prediction Overview
This is a package that been created to train and validate a machine 
learning model, to predict the target values of the dataset, after applying 
a series of preprocessing steps on the raw data.

Analytically, two modules have been created:
 - Preprocessing, which includes:
    - Data Exploration
    - Data Cleansing
    - Data Merging
    - Feature engineering
    - Feature selection
    - Data transformation

 - Modeling, which includes:
    - Implementation of a Random Forest regression model
    - Randomized search for hyperparameter tuning
    - Training of the model
    - Evaluation of the model


### Requirements
Install tox using the command `pip install tox`.

### Environment creation
The environment can be created with the command `tox -e develop`.

### Code execution
The code can be execute by running the `main.py` file.

The initial csv data `md_target_dataset.csv` and
 `md_raw_dataset.csv` must be located under the `raw_data` folder.

In the `config.py` file, the data types of the both the 
Raw and Target datasets' columns can be specified, if deemed necessary.
The columns can be categorized into 
Datetime, Integer, Float and Categorical data types.
                           
In the Data Preprocessing step, the user can select whether or not to run 
initial Automatic Profiling and whether or not to generate 
Exploratory Analysis Plots, by changing the respective boolean arguments
`auto_profiling` and `generate_plots`.

Example:

    df_preprocessed = dp.preprocess_dfs(auto_profiling=False,
                                        generate_plots=True)

In the Data Modelling step, the user can select whether or not to perform
randomized search for hyperparameter tuning, by changing the
 boolean arguments `random_search`.
 
Example:

    data_modeling.random_forest(random_search=True)

### Tests
Under the `tests` folder can be found some *indicative* tests.

The tests can be run with the command `python -m pytest`.

### Additional information
The plots generated are being saved under the `plots` folder.

The initial generated profiling html files can be found under the 
`profiling` folder.