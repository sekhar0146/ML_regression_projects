"""
Predicting the sale price of Bulldozers using Machine learing 

Train.csv is the training set, which contains data through the end of 2011.

Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 
You make predictions on this set throughout the majority of the competition. 
Your score on this set is used to create the public leaderboard.

Test.csv is the test set, which won't be released until the last week of the competition. 
It contains data from May 1, 2012 - November 2012. 
Your score on the test set determines your final rank for the competition.
The key fields are in train.csv are:

SalesID: the uniue identifier of the sale
MachineID: the unique identifier of a machine.  A machine can be sold multiple times
saleprice: what the machine sold for at auction (only provided in train.csv)
saledate: the date of the sale

Evaluation: The evaluation metric for this competition is the 
RMSLE (root mean squared log error) between the actual and 
predicted auction prices.
The goal is to minimise the RMSLE between actual and predicted price.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.ensemble import RandomForestRegressor

# Evaluation
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

# import training and validation sets
df = pd.read_csv("C:/Users/z011348/Desktop/ML/input/bluebook-for-bulldozers/TrainAndValid.csv",
                 low_memory=False)
#print(df.head())
#print(df.info())
# check for null/missing data
#print("====== Missing data information ======")
#print(df.isna().sum())
#print(df.saledate.dtype)
#print(df.saledate[:10])
# plot the data between saledate and Price
#fig1, ax1 = plt.subplots()
#ax1.scatter(df["saledate"][:1000], df["SalePrice"][:1000])

# craete histogram on saleprice
#df.SalePrice.plot.hist();

"""
Parsing dates
when we work with time series data, we want to enrich 
the date and time as much as possible. 

we can do that by telling pandas which of our columns has 
dates in it using 'parse_dates' parameter 
"""

# import the data again with parse dates
df = pd.read_csv("C:/Users/z011348/Desktop/ML/input/bluebook-for-bulldozers/TrainAndValid.csv",
                 low_memory=False,
                 parse_dates=["saledate"]) # convert into yyyy-mm-dd
print(df.saledate.dtype)
print(df.saledate[:10])
# plot the data between saledate and Price
fig1, ax1 = plt.subplots()
ax1.scatter(df["saledate"][:1000], df["SalePrice"][:1000])

# Sort the DataFrame by saledate
# when working with time series, it is good to sort it by date
df.sort_values(by=["saledate"], 
               inplace=True,
               ascending=True)
print(df.saledate[:4])

# Make a copy of original dataframe - for future reference
df_tmp = df.copy()

# add datetime parameter for saledate column
df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear
print(df_tmp.head())

# Now we have enrichied our dataframe with date time features,
# we can remove saledate
print("")
df_tmp.drop("saledate", axis=1, inplace=True)

"""
SPlit the data
"""
#X = df_tmp.drop("SalePrice", axis=1)
#y = df_tmp["SalePrice"]

"""
MODELING:
=========
Lets build a machine learning model
"""
# model = RandomForestRegressor(n_jobs=-1,
#                              random_state=42)
# will get an error because we have not converted string cols to numeric
# model.fit(X, y)

"""
Convert Strings to categories
One way we can turn all our data into number is by converting them
into Pandas categories
"""
print("=== Convert Strings to Pandas categories ===")
# find columns which contains strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
       # print(label)
        df_tmp[label] = content.astype("category").cat.as_ordered()
print(df_tmp.info())
print("")
#print(df_tmp.state.cat.categories)       
#print(df_tmp.state.cat.codes)      
#print(df_tmp.state[:10])

# Save preprocessed data to aviod the decimal values for int
df_tmp.to_csv("C:/Users/z011348/Desktop/ML/input/bluebook-for-bulldozers/train_tmp.csv",
              index=False)
# import the tmp data
df_tmp = pd.read_csv("C:/Users/z011348/Desktop/ML/input/bluebook-for-bulldozers/train_tmp.csv")
#print(df_tmp.isna().sum())

"""
FIll missing values
Fill numeric missing values first
"""
print("=== FIll missing numeric values ===")
# check which numeric columns have missing values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        #x = pd.isnull(content).sum()
        #print(x)
        if pd.isnull(content).sum():
            # add binary column which tells us if the data was missing or not
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with meadian
            df_tmp[label] = content.fillna(content.median())

# check for the data
print(df_tmp.auctioneerID_is_missing.value_counts())

# ----------------------------------------------------------
# Fill categorical variables into numbers and fill missing
# ----------------------------------------------------------
# check for columns whihc are not numeric
print("=== Fill categorical variables into numbers ===")
#print(pd.Categorical(df_tmp["state"]))
#print("")
#print(pd.Categorical(df_tmp["state"]).codes)

for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # print(label)
        # add a binary col to indicate whether sample has missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # turn categories into numbers and +1
        # if missing value is there for Categorical the value will be -1
        # So, to aviod to have -1 we are adding 1
        df_tmp[label] = pd.Categorical(content).codes+1

print("")
print(df_tmp.info())
print("=== check for missing values")
print(df_tmp.isna().sum())
print("")
"""
AGAIN MODELING:
=========
Lets build a machine learning model
"""
print("=== Modeling === ")

X = df_tmp.drop("SalePrice", axis=1)
y = df_tmp["SalePrice"]

#model = RandomForestRegressor(n_jobs=-1,
#                              random_state=42)
#model.fit(X, y)

# score
#print(model.score(X, y))

## ==> The above metric is not reliable because 
# we have processed data which is not trained/test

# Split the data into train and validation sets
# As per the kaggle, 2012 - valid set, till 2011 - train set
# print(df_tmp.saleYear.value_counts())
print("=== Split the data into train and validation sets")
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]
print(len(df_val), len(df_train))

# Split into X and y (on train and valid sets)
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train["SalePrice"]
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val["SalePrice"]

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

# -------------------------------
# Building an evaluation function
# -------------------------------
# Create evaluation function  (RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_pred):
    """
    calculates room mean squared log error b/w true and prediction values
    """
    return np.sqrt(mean_squared_log_error(y_test, y_pred))

# create a function on evaluate model on few different levels
def show_scores(model):
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_pred),
              "Valid MAE": mean_absolute_error(y_valid, val_pred),
              "Training RMSLE" : rmsle(y_train, train_pred),
              "Valid RMSLE": rmsle(y_valid, val_pred),
              "Training R^2": r2_score(y_train, train_pred),
              "Valid R^2": r2_score(y_valid, val_pred)
              }
    return scores

# -----------------------------------------
# Testing our model on subset (To tune the hyperparameters)
# -----------------------------------------
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000) # try for 10k records
model.fit(X_train, y_train)
print(show_scores(model))
print("")
# ------------------------------------------------------------
# Hyperparameter tuning with RandomizedSearchCV
# ------------------------------------------------------------
print("=== Hyperparameter tuning with RandomizedSearchCV ===")
from sklearn.model_selection import RandomizedSearchCV
# different RandomForestRegressor hyperparameters
rf_grid = {"n_estimators": np.arange(10, 100, 10),
           "max_depth": [None, 3 , 5, 10],
           "min_samples_split":np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2),
           "max_features": [0.5, 1, "sqrt", "auto"],
           "max_samples": [10000]}

# Instantiate RandomizedSearchCV model
rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                    random_state=42),
                              param_distributions=rf_grid,
                              n_iter=2,
                              cv=5,
                              verbose=True)
rs_model.fit(X_train, y_train)
print("Hyperparameter best parameters")
print(rs_model.best_params_)
print(show_scores(rs_model))
print("")   
# -------------------------------------------
# Train a model with the best Hyperparameters (by hand) trained on all data
# -------------------------------------------
# most ideal Hyperparameters
print("===Train a model with the best Hyperparameters (by hand) trained on all data")
ideal_model = RandomForestRegressor(n_estimators=40,
                                    min_samples_leaf=1,
                                    min_samples_split=14,
                                    max_features=0.5,
                                    n_jobs=-1,
                                    max_samples=None,
                                    random_state=42)
ideal_model.fit(X_train, y_train)
print(show_scores(ideal_model))

# -------------------------------------------------
# Make predictions on test data set
# -------------------------------------------------
# import the data again with parse dates
df_test = pd.read_csv("C:/Users/z011348/Desktop/ML/input/bluebook-for-bulldozers/Test.csv",
                 low_memory=False,
                 parse_dates=["saledate"])

# print(df_test.head())
# we will get an error while pedicting it because we have missing data and non-numeric
#test_pred = ideal_model.predict(df_test)

# ****
# Preprocessing the data ( getting the test data set in same format in train set)
# ****
def preprocess_date(df):
    # add datetime parameter for saledate column
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    
    df.drop("saledate", axis=1, inplace=True)
    
    # Fill numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
        #x = pd.isnull(content).sum()
        #print(x)
            if pd.isnull(content).sum():
            # add binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with meadian
                df[label] = content.fillna(content.median())
            
        # Fill catogortical missing data and turn into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            # we add +1 to category code
            df[label] = pd.Categorical(content).codes+1
    
    return df
    
# Process test data
df_test = preprocess_date(df_test)
print(df_test.head())
print("")
# Find the column differences b/w training and test data sets
print("Find the column differences b/w training and test data sets:")
print(set(X_train.columns) - set(df_test.columns)) # auctionerID_is_missing
print("")

# manually adjust df_test to have auctionerID_is_missing column 
print("manually adjust df_test to have auctionerID_is_missing column:")
df_test["auctionerID_is_missing"] = False
print(df_test.head())
print("")
# Now we have same columns at df_test and X_train. Lets do prediction
print("========= Predicted salePrice ============ ")
test_pred = ideal_model.predict(df_test)
print(test_pred)
print("")

# We have made some predictions but they are not in the same as Kaggle
# Now we need to format as per the Kaggle format
df_prediction_saleprice = pd.DataFrame()
df_prediction_saleprice["SalesID"] = df_test["SalesID"]
df_prediction_saleprice["SalePrice"] = test_pred
print(df_prediction_saleprice)

# Save the prediction results in the csv file
df_prediction_saleprice.to_csv("C:/Users/z011348/Desktop/ML/output/bluebook-for-bulldozers/price_predictions.csv",
                               index=False)