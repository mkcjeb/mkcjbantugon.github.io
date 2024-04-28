---
title: "Chicago Bike Rental: Regression-based Analysis"
date: 2024-02-28
tags: [Python, machine learning, regression]
header:
  image: "/images/bike.jpg"
excerpt: "(Python - Machine Learning) This regression analysis and model aims to developing a machine learning model to predict the number of bike rentals on a given day, as well as to provide insights into the factors that contribute to bike rental demand. 
Business case built by Professor Chase Kusterer from Hult International Business School"
mathjax: "true"
toc: true
toc_label : "Navigate"
---
By: Michelle Kae Celine Jo-anne Bantugon<br>

Business case built by Professor Chase Kusterer<br>
Hult International Business School<br><br>
---------------------------------------------------------------------------------------------------
This regression analysis and model aims to developing a machine learning model to predict the number of bike rentals on a given day, as well as to provide insights into the factors that contribute to bike rental demand. <br>

Jupyter notebook and dataset for this analysis can be found here: [Portfolio-Projects](https://github.com/sbriques/portfolio-projects)

***
### Introduction

The bike sharing industry has grown tremendously in recent years, with an estimated global value of $2.8 billion in 2023. This is due to a number of factors, such as convenience, sustainability, and physical fitness. As a result of the market's growth, accurately predicting accurate bike rentals presents a significant challenge due to the dynamic nature of bikesharing systems and the unpredictable behavior of riders. These challenges result in fluctuating demand, leading to inefficiencies in operation, decreased service levels, and potential user dissatisfaction.
<br>
Therefore, it is essential to understand future demand patterns that can help reduce relocation costs and improve system performance.

### Overview

- Best performing model after tuning was a Lasso Regression with 27 features with a test score of 0.7688 and a cross validation score with 5 folds with train-test gap of 0.0198
- Small alpha was used and cyclic selection
-  Used of L1 penalty to shrink the coefficients of less important features to zero, effectively performing feature selection.
***

<strong> Case - Chicago Bike Rental. </strong> <br>
<strong>  Audience: </strong> Cook County Planning and Development Department  <br>
<strong> Goal: </strong> To predict the number of bike rentals on a given day, as well as to provide insights into the factors that contribute to bike rental demand. <br>

***

<strong> Analysis Outline: </strong>
1. Part 1: Exploratory Data Analysis
2. Part 2: Transformations
3. Part 3: Build a machine learning model to predict bike rentals 
4. Part 4: Evaluating Model

*** 

### Libraries and Packages


```python
# Importing libraries
import numpy  as np                                    # mathematical essentials
import pandas as pd                                    # data science essentials
import matplotlib.pyplot as plt                        # essential graphical output
import seaborn as sns                                  # enhanced graphical output
import statsmodels.formula.api as smf                  # regression modeling
import sklearn.linear_model                            # linear models
from sklearn.model_selection import train_test_split   # train/test split
from sklearn.preprocessing import StandardScaler       # standard scaler
from sklearn.neighbors import KNeighborsRegressor      # KNN for Regression
from sklearn.tree import DecisionTreeRegressor         # regression trees
from sklearn.tree import plot_tree                     # tree plots
from sklearn.model_selection import RandomizedSearchCV # hyperparameter tuning
import warnings

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# suppressing warnings
warnings.filterwarnings(action = 'ignore')
```

### Kaggle Dataset

```python
## importing data ##

# reading modeling data into Python
modeling_data = './datasets/train.xlsx'

# calling this df_train
df_train = pd.read_excel(io         = modeling_data,
                         sheet_name = 'data',
                         header     = 0,
                         index_col  = 'ID')

# reading testing data into Python
testing_data = './datasets/test.xlsx'

# calling this df_test
df_test = pd.read_excel(io         = testing_data,
                        sheet_name = 'data',
                        header     = 0,
                        index_col  = 'ID')
# concatenating datasets together for mv analysis and feature engineering
df_train['set'] = 'Not Kaggle'
df_test ['set'] = 'Kaggle'

# concatenating both datasets together for mv and feature engineering
df_full = pd.concat(objs = [df_train, df_test],
                    axis = 0,
                    ignore_index = False)

# checking data
df_full.head(n = 5)

```

***

## Setting the response variable
```

#!##############################!#
#!# set your response variable #!#
#!##############################!#
y_variable = 'RENTALS' # for OLS, KNN, Ridge, Lasso,SGD, and Decision Tree

```

## Part I: Base Modeling

```python
## Base Modeling ##

# Step 1: INSTANTIATE a model object
lm_base = smf.ols(formula = """RENTALS ~ Q('Temperature(F)') +
                                       Q('Humidity(%)') +
                                       Q('Visibility(miles)') +
                                       Q('DewPointTemperature(F)') +
                                       Q('Rainfall(in)') +
                                       Q('Snowfall(in)') +
                                       Q('SolarRadiation(MJ/m2)') +
                                       Q('Wind speed (mph)') +
                                       Holiday +
                                       FunctioningDay""",
                                       data = df_full)

# Step 2: FIT the data into the model object
results = lm_base.fit()

# Step 3: analyze the SUMMARY output
print(results.summary())

```
## Part II: Exploratory Data Analysis (EDA)

Missing Value Analysis and Imputation

```
# Checking missing values excluding 'RENTALS' 
df_full.iloc[:, :-2].isnull().sum(axis=0)
```

Transformation

```
# TRANSFORMATION
# Remove milliseconds from the 'DateHour' column since all did not have ms
df_full['DateHour'] = df_full['DateHour'].str.split('.').str[0]

# convert the 'DateHour' column to datetime format
df_full['DateHour'] = df_full['DateHour'].astype('datetime64[ns]')

# Round hour if minute is 59
df_full.loc[df_full['DateHour'].dt.minute == 59, 'DateHour'] += pd.Timedelta(minutes=1)
df_full['DateHour'] = df_full['DateHour'].dt.floor('H')

# Extract Day, Month, and Hour
df_full['Day'] = df_full['DateHour'].dt.day_name()
df_full['Month'] = df_full['DateHour'].dt.month
df_full['Hour'] = df_full['DateHour'].dt.hour

# Convert Month and Hour to integers
df_full['Month'] = df_full['Month'].astype(int)
df_full['Hour'] = df_full['Hour'].astype(int)
```

Imputation

```
## IMPUTATION

# Calculate median dew point temperature per month
median_dewpoint_per_month = df_full.groupby('Month')['DewPointTemperature(F)'].median()

# Fill missing values in each month with the corresponding median dew point temperature
df_full.loc[df_full['Month'] == 9, 'DewPointTemperature(F)'] = df_full.loc[df_full['Month'] == 9, 'DewPointTemperature(F)'].fillna(median_dewpoint_per_month[9])
df_full.loc[df_full['Month'] == 10, 'DewPointTemperature(F)'] = df_full.loc[df_full['Month'] == 10, 'DewPointTemperature(F)'].fillna(median_dewpoint_per_month[10])
df_full.loc[df_full['Month'] == 11, 'DewPointTemperature(F)'] = df_full.loc[df_full['Month'] == 11, 'DewPointTemperature(F)'].fillna(median_dewpoint_per_month[11])
df_full.loc[df_full['Month'] == 12, 'DewPointTemperature(F)'] = df_full.loc[df_full['Month'] == 12, 'DewPointTemperature(F)'].fillna(median_dewpoint_per_month[12])

# Calculate median visibility per month
median_visibility_per_month = df_full.groupby('Month')['Visibility(miles)'].median()

# Calculate median solar radiation per month
median_solar_radiation_per_month = df_full.groupby('Month')['SolarRadiation(MJ/m2)'].median()

# Fill missing values in each month with the corresponding median visibility
df_full.loc[df_full['Month'] == 9, 'Visibility(miles)'] = df_full.loc[df_full['Month'] == 9, 'Visibility(miles)'].fillna(median_visibility_per_month[9])
df_full.loc[df_full['Month'] == 10, 'Visibility(miles)'] = df_full.loc[df_full['Month'] == 10, 'Visibility(miles)'].fillna(median_visibility_per_month[10])
df_full.loc[df_full['Month'] == 11, 'Visibility(miles)'] = df_full.loc[df_full['Month'] == 11, 'Visibility(miles)'].fillna(median_visibility_per_month[11])
df_full.loc[df_full['Month'] == 12, 'Visibility(miles)'] = df_full.loc[df_full['Month'] == 12, 'Visibility(miles)'].fillna(median_visibility_per_month[12])

# Fill missing values in each month with the corresponding median solar radiation
df_full.loc[df_full['Month'] == 9, 'SolarRadiation(MJ/m2)'] = df_full.loc[df_full['Month'] == 9, 'SolarRadiation(MJ/m2)'].fillna(median_solar_radiation_per_month[9])
df_full.loc[df_full['Month'] == 10, 'SolarRadiation(MJ/m2)'] = df_full.loc[df_full['Month'] == 10, 'SolarRadiation(MJ/m2)'].fillna(median_solar_radiation_per_month[10])
df_full.loc[df_full['Month'] == 11, 'SolarRadiation(MJ/m2)'] = df_full.loc[df_full['Month'] == 11, 'SolarRadiation(MJ/m2)'].fillna(median_solar_radiation_per_month[11])
df_full.loc[df_full['Month'] == 12, 'SolarRadiation(MJ/m2)'] = df_full.loc[df_full['Month'] == 12, 'SolarRadiation(MJ/m2)'].fillna(median_solar_radiation_per_month[12])
```

Transformation

```
# TRANSFORMATION
# Remove milliseconds from the 'DateHour' column since all did not have ms
df_full['DateHour'] = df_full['DateHour'].str.split('.').str[0]

# convert the 'DateHour' column to datetime format
df_full['DateHour'] = df_full['DateHour'].astype('datetime64[ns]')

# Round hour if minute is 59
df_full.loc[df_full['DateHour'].dt.minute == 59, 'DateHour'] += pd.Timedelta(minutes=1)
df_full['DateHour'] = df_full['DateHour'].dt.floor('H')

# Extract Day, Month, and Hour
df_full['Day'] = df_full['DateHour'].dt.day_name()
df_full['Month'] = df_full['DateHour'].dt.month
df_full['Hour'] = df_full['DateHour'].dt.hour

# Convert Month and Hour to integers
df_full['Month'] = df_full['Month'].astype(int)
df_full['Hour'] = df_full['Hour'].astype(int)
```

## Part III. Feature Engineering

```
## Feature Engineering ##

##############################################################################
## Feature 1 Temperature and Functioning Day Interaction

# Temperature x FunctionIng Day Interaction
df_full['Temp_FunctioningDay'] = df_full['Temperature(F)'] * df_full['FunctioningDay']

##############################################################################
## Feature 2 Humidity and Dew Point Temperature Interaction

# Humidity x Functioning Day Interaction
df_full['HumidityFDay'] = df_full['Humidity(%)'] * df_full['DewPointTemperature(F)']
    
##############################################################################
## Feature 3 Precipitation

df_full['Precipitation'] = (df_full['Rainfall(in)'] + df_full['Snowfall(in)'] > 0).astype(int)
    
##############################################################################
## Feature 4 Heat Index
## Based on 1990 National Weather Service (NWS) Formula

# Defining a function to calculate the Heat Index
def calculate_heat_index(row):
    temperature = row['Temperature(F)']
    humidity = row['Humidity(%)']
    
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6

    heat_index = (
        c1 +
        c2 * temperature +
        c3 * humidity +
        c4 * temperature * humidity +
        c5 * temperature**2 +
        c6 * humidity**2 +
        c7 * temperature**2 * humidity +
        c8 * temperature * humidity**2 +
        c9 * temperature**2 * humidity**2
    )

    return round(heat_index, 2)

# Applying the function to create a new column 'Heat_Index'
df_full['Heat_Index'] = df_full.apply(calculate_heat_index, axis=1)

# Displaying the results
df_full.head(n=3)
```
## OLS Modeling with New Features
```
## Modeling with New Features ##

# Step 1: INSTANTIATE a model object
lm_best = smf.ols(formula =  """RENTALS  ~  
                                            Hour_1 +
                                            Hour_2 +
                                            Hour_3 +
                                            Hour_4 +
                                            Hour_5 +
                                            Hour_6 +
                                            Hour_7 +
                                            Hour_8 +
                                            Hour_9 +
                                            Hour_15 +
                                            Hour_16 +
                                            Hour_17 +
                                            Hour_18 +
                                            Hour_19 +
                                            Hour_20 +
                                            Hour_21 +
                                            Hour_22 +
                                            Saturday +
                                            Sunday +
                                            Wednesday +
                                            Q('DewPointTemperature(F)') +
                                            HumidityFDay +
                                            Temp_FunctioningDay +
                                            Precipitation +
                                            Q('Wind speed (mph)') +
                                            Holiday + 
                                            Heat_Index""",
                                            data = df_full)

# Step 2: FIT the data into the model object
results = lm_best.fit()

# Step 3: analyze the SUMMARY output
print(results.summary())
```

## Part IV. Data Partitioning
Separating the Kaggle Data

```
## parsing out testing data (needed for later) ##

# dataset for kaggle
kaggle_data = df_full[ df_full['set'] == 'Kaggle' ].copy()


# dataset for model building
df = df_full[ df_full['set'] == 'Not Kaggle' ].copy()


# dropping set identifier (kaggle)
kaggle_data.drop(labels = 'set',
                 axis = 1,
                 inplace = True)


# dropping set identifier (model building)
df.drop(labels = 'set',
        axis = 1,
        inplace = True)
```

Standardization

```
x_data  = df.drop(['DateHour'],
                          axis = 1)

## Standardization ##

# # INSTANTIATING a StandardScaler() object
scaler = StandardScaler()


# # FITTING and TRANSFORMING
x_scaled = scaler.fit_transform(x_data)


# # converting scaled data into a DataFrame
x_scaled_df = pd.DataFrame(x_scaled)


# # labeling columns
x_scaled_df.columns = x_data.columns


x_data = x_scaled_df.copy()

# # checking the results
x_data.describe(include = 'number').round(decimals = 2)

```

Train-Test Split

```
#!###########################!#
#!# choose your x-variables #!#
#!###########################!#
x_features = [  
                                            'Hour_1',
                                            'Hour_2',
                                            'Hour_3',
                                            'Hour_4',
                                            'Hour_5',
                                            'Hour_6',
                                            'Hour_7',
                                            'Hour_8',
                                            'Hour_9',
                                            'Hour_15',
                                            'Hour_16',
                                            'Hour_17',
                                            'Hour_18',
                                            'Hour_19',
                                            'Hour_20',
                                            'Hour_21',
                                            'Hour_22',
                                            'Saturday',
                                            'Sunday',
                                            'Wednesday',
                                            'DewPointTemperature(F)',
                                            'HumidityFDay',
                                            'Temp_FunctioningDay',
                                            'Precipitation',
                                            'Wind speed (mph)', 
                                            'Heat_Index',
                                            'Holiday']
```

```
## ########################### ##
## DON'T CHANGE THE CODE BELOW ##
## ########################### ##

# prepping data for train-test split
y_data = df[y_variable]


# removing non-numeric columns and missing values
x_data = df[x_features].copy().select_dtypes(include=[int, float]).dropna(axis = 1)


# storing remaining x_features after the step above
x_features = list(x_data.columns)


# train-test split (to validate the model)
x_train, x_test, y_train, y_test = train_test_split(x_data, 
                                                    y_data, 
                                                    test_size    = 0.25,
                                                    random_state = 702 )


# results of train-test split
print(f"""
Original Dataset Dimensions
---------------------------
Observations (Rows): {df.shape[0]}
Features  (Columns): {df.shape[1]}


Training Data (X-side)
----------------------
Observations (Rows): {x_train.shape[0]}
Features  (Columns): {x_train.shape[1]}


Training Data (y-side)
----------------------
Feature Name:        {y_train.name}
Observations (Rows): {y_train.shape[0]}


Testing Data (X-side)
---------------------
Observations (Rows): {x_test.shape[0]}
Features  (Columns): {x_test.shape[1]}


Testing Data (y-side)
---------------------
Feature Name:        {y_test.name}
Observations (Rows): {y_test.shape[0]}""")
```

## Part V. Candidate Modeling

```
## Candidate Modeling ##

# List of models to loop through (OLS, Ridge, Lasso, SGD)
models = [
    ('Linear Regression (log Rentals)', sklearn.linear_model.LinearRegression()),
    ('Ridge (Not Tuned)', sklearn.linear_model.Ridge(alpha = 0.01, random_state = 702)),
    ('Lasso (Not Tuned)', sklearn.linear_model.Lasso(alpha = 0.0001, random_state = 702)),
    ('SGD (Not Tuned)',   sklearn.linear_model.SGDRegressor(alpha = 2, random_state = 702, penalty = 'elasticnet',\
                                                            loss = 'epsilon_insensitive', learning_rate = 'adaptive'))
]

# Placeholder DataFrame to store coefficients
coef_df = pd.DataFrame(columns=['Model Name', 'train_RSQ', 'test_RSQ', 'tt_gap', 'Intercept'] + list(x_data.columns))

for model_name, model in models:
    # FITTING to the training data
    model_fit = model.fit(x_train, y_train)

    # PREDICTING on new data
    model_pred = model.predict(x_test)

    # SCORING the results
    model_train_score = model.score(x_train, y_train).round(4)
    model_test_score = model.score(x_test, y_test).round(4)
    model_gap = abs(model_train_score - model_test_score).round(4)

    # Setting up a placeholder list to store model features
    coefficient_lst = [('intercept', model.intercept_.round(decimals=4))]

    # Printing out each feature-coefficient pair one by one
    for coefficient in model.coef_.round(decimals=4):
        coefficient_lst.append(coefficient)

    # Instantiating a list to store model results
    coef_lst = [model_name, model_train_score, model_test_score, model_gap, model.intercept_.round(decimals=6)]

    # Extending list with feature coefficients
    coef_lst.extend(model.coef_.round(decimals=6))

    # Converting to DataFrame
    coef_lst = pd.DataFrame(data=coef_lst)

    # Transposing (rotating) DataFrame
    coef_lst = np.transpose(coef_lst)

    # Adding column names
    coef_columns = ['Model Name', 'train_RSQ', 'test_RSQ', 'tt_gap', 'Intercept']
    coef_columns.extend(x_data.columns)
    coef_lst.columns = coef_columns

    # Concatenating to coef_df
    coef_df = pd.concat(objs=[coef_df, coef_lst], axis=0, ignore_index=True)

# Displaying the results
coef_df
```

for KNN
```
## for KNN
model_name = 'KNN (Not Tuned)' # name your model

# model type
model = KNeighborsRegressor(algorithm = 'auto',
                            n_neighbors = 5)

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4) # using R-square
model_test_score  = model.score(x_test, y_test).round(4)   # using R-square
model_gap         = abs(model_train_score - model_test_score).round(4)
    
# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""
print(model_summary)
```

for Decision Tree
```
## for Decision Tree
model_name = 'Unpruned Regression Tree' # name your model

# model type
model = DecisionTreeRegressor(min_samples_leaf = 9,
                              max_depth        = 6,
                              random_state     = 702)

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4) # using R-square
model_test_score  = model.score(x_test, y_test).round(4)   # using R-square
model_gap         = abs(model_train_score - model_test_score).round(4)
    
# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""
print(model_summary)
```

## Residual Analysis
```
# Plotting Residual Analysis per model

# List of models to loop through
models = [
    ('OLS (log Rentals)', sklearn.linear_model.LinearRegression()),
    ('Ridge (Not Tuned)', sklearn.linear_model.Ridge(alpha=0.01, random_state=702)),
    ('Lasso (Not Tuned)', sklearn.linear_model.Lasso(alpha=0.01, random_state=702)),
    ('KNN (Not Tuned)', KNeighborsRegressor(algorithm='auto', n_neighbors=6)),
    ('SGD (Not Tuned)', sklearn.linear_model.SGDRegressor(alpha=2, random_state=702, penalty='elasticnet',\
                                                          loss='epsilon_insensitive', learning_rate='adaptive')),
    ('Decision Tree (Not Tuned)', DecisionTreeRegressor(min_samples_leaf = 9, max_depth = 6, random_state = 702))
]
# Create subplots for each model
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

for i, (model_name, model) in enumerate(models):
    # FITTING to the training data
    model_fit = model.fit(x_train, y_train)

    # PREDICTING on new data
    model_pred = model.predict(x_test)

    # organizing residuals
    model_residuals = {"True": y_test, "Predicted": model_pred}

    # converting residuals into df
    model_resid_df = pd.DataFrame(data=model_residuals)

    # developing a residual plot
    ax = axs[i // 3, i % 3]
    sns.residplot(data=model_resid_df, x='Predicted', y='True', lowess=True, color='blue',
                  scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'}, ax=ax)
    
    # title and axis labels
    ax.set_title(f"Residual Plot - {model_name}")
    ax.set_xlabel("Predicted Bike Rental (log)")
    ax.set_ylabel("Actual Bike Rental (log)")

# layout and rendering visual
plt.tight_layout()
plt.show()
```
Candidate Model Development and Final Model Selection
1. Decision Tree
Decision Tree was chosen to have a baseline of the model since it is nonparametric model type and assume no model form. Transformation is not needed and it can help generate useful information for developing hypotheses and creating other model as well.It captures nonlinear relationships and interactions among variables, suitable for complex patterns in bike rental data.
2. Ridge Regression
3. Lasso Regression

The x_data is scaled and it shows no difference in the model score among the three models. The model performance was countercheck in Kaggle to see if there will be any significant difference in the r square and it shows minimal difference. There r square are near each other after tuning which means it can predict on new unseen data after training and tuning.

Ridge and Lasso Regression It helps reduce overfitting by adding a penalty to the regression coefficients, promoting simpler models. Ridge adds a penalty equal to the square of the magnitude of coefficients, while Lasso adds a penalty equal to the absolute value of coefficients.

## Part VI. Hyperparameter Tuning
Lasso 
```
## Hyperparameter Tuning ##

## Lasso
model_name = 'Lasso (Not Tuned)'


# INSTANTIATING model object
model = sklearn.linear_model.Lasso(alpha         = 0.0001,
                                   selection     = 'random',
                                   random_state  = 702)

## ########################### ##
## DON'T CHANGE THE CODE BELOW ##
## ########################### ##

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)
```

```
## Tuning Lasso Model

# declaring a hyperparameter space
#alpha_range = (0.00001, 0.0001, 0.001, 0.01, 1, 1.5, 2, 2.5, 3)
#fit_range   = [True, False]
#max_range   = np.arange(1000, 11000, 1000)
#selection_range = ['cyclic', 'random']

# creating a hyperparameter grid
#param_grid = {'alpha'         : alpha_range,
#              'fit_intercept' : fit_range,
#              'max_iter'      : max_range,
#              'selection'     : selection_range}

# INSTANTIATING the model object without hyperparameters
#tuned_lasso = sklearn.linear_model.Lasso(random_state = 219)


# RandomizedSearchCV object
#tuned_lasso_cv = RandomizedSearchCV(estimator            = tuned_lasso, # model
#                                   param_distributions   = param_grid,  # hyperparameter ranges
#                                   cv                    = 5,           # folds 
#                                   n_iter                = 1000,        # how many models to build
#                                   random_state          = 702)


# FITTING to the FULL DATASET (due to cross-validation)
#tuned_lasso_cv.fit(x_data, y_data)


# printing the optimal parameters and best score
#print("Tuned Parameters  :", tuned_lasso_cv.best_params_)
#print("Tuned Training R-squared:", tuned_lasso_cv.best_score_.round(4))
# Tuned Parameters  : {'selection': 'cyclic', 'max_iter': 1000, 'fit_intercept': True, 'alpha': 0.01}
# Tuned Training R-squared: 0.7499
```

```
# Tuning Results
# Tuned Parameters  : {'selection': 'cyclic', 'max_iter': 1000, 'fit_intercept': True, 'alpha': 0.01}
# Tuned Training R-squared: 0.7499

# New Prediction using tuning parameters result

## Lasso
model_name = 'Lasso (Tuned)'


# INSTANTIATING model object
model = sklearn.linear_model.Lasso(alpha         = 0.01,
                                   selection     = 'cyclic',
                                   max_iter      = 1000,
                                   fit_intercept = True,
                                   random_state  = 702)

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)
# Model Name:     Lasso (Tuned)
# Train_Score:    0.7688
# Test_Score:     0.749
# Train-Test Gap: 0.0198
```
Ridge Regression Model
```
## Hyperparameter Tuning ##

## Ridge
model_name = 'Ridge (Not Tuned)'

# INSTANTIATING a model object
model = sklearn.linear_model.Ridge(alpha         = 0.01,
                                   random_state  = 702)

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)
```
```
# Tuning Ridge Model

# declaring a hyperparameter space
#alpha_range = (0.00001, 0.0001, 0.001, 0.01, 1, 1.5, 2, 2.5, 3)
#fit_range   = [True, False]
#max_range   = np.arange(1000, 11000, 1000)
#solver_range = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
#positive_range = [True, False]



# creating a hyperparameter grid
#param_grid = {'alpha'         : alpha_range,
#              'fit_intercept' : fit_range,
#              'max_iter'      : max_range,
#              'solver'        : solver_range,
#              'positive'      : positive_range}



# INSTANTIATING the model object without hyperparameters
#tuned_ridge = sklearn.linear_model.Ridge(random_state = 219)


# RandomizedSearchCV object
#tuned_ridge_cv = RandomizedSearchCV(estimator            = tuned_ridge, # model
#                                   param_distributions   = param_grid,  # hyperparameter ranges
#                                   cv                    = 5,           # folds 
#                                   n_iter                = 1000,        # how many models to build
#                                   random_state          = 702)


# FITTING to the FULL DATASET (due to cross-validation)
#tuned_ridge_cv.fit(x_data, y_data)


# printing the optimal parameters and best score
#print("Tuned Parameters  :", tuned_ridge_cv.best_params_)
#print("Tuned Training R-squared:", tuned_ridge_cv.best_score_.round(4))
```
```
# Tuning Results

#Tuned Parameters  : {'solver': 'sparse_cg', 'positive': False, 'max_iter': 6000, 'fit_intercept': True, 'alpha': 3}
#Tuned Training R-squared: 0.95


# New Prediction using tuning parameters result

## Ridge
model_name = 'Ridge (Tuned)'


# INSTANTIATING a model object - CHANGE THIS AS NEEDED
model = sklearn.linear_model.Ridge(alpha         = 3,
                                   solver        = 'sparse_cg',
                                   positive      = False,
                                   max_iter      = 6000,
                                   fit_intercept = True,
                                   random_state  = 702)

# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)
```
Decision Tree
```
## Hyperparameter Tuning ##

## Decision Tree
model_name = 'Unpruned Regression Tree (Not Tuned)'

# INSTANTIATING a model object
model = DecisionTreeRegressor(min_samples_leaf = 9,
                              max_depth        = 6,
                              random_state     = 702)

# FITTING to the training data
model_fit = model.fit(x_train, y_train)

# PREDICTING on new data
model_pred = model.predict(x_test)

# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    
# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)
```
```
# Tuning Decision Tree Regression Model

# declaring a hyperparameter space
#criterion_range = ["squared_error", "friedman_mse", "absolute_error", "poisson"]
#splitter_range  = ["best", "random"]
#depth_range     = np.arange(1,11, 1)
#leaf_range      = np.arange(1,251, 5)


# creating a hyperparameter grid
#param_grid = {'criterion'        : criterion_range,
#             'splitter'          : splitter_range,
#             'max_depth'         : depth_range,
#             'min_samples_leaf'  : leaf_range}

# INSTANTIATING the model object without hyperparameters
#tuned_tree = DecisionTreeRegressor(random_state = 219)


# RandomizedSearchCV object
#tuned_tree_cv = RandomizedSearchCV(estimator             = tuned_tree, # model
#                                   param_distributions   = param_grid, # hyperparameter ranges
#                                   cv                    = 5,          # folds 
#                                   n_iter                = 1000,       # how many models to build
#                                  random_state          = 702)


# FITTING to the FULL DATASET (due to cross-validation)
#tuned_tree_cv.fit(x_data, y_data)


# printing the optimal parameters and best score
#print("Tuned Parameters  :", tuned_tree_cv.best_params_)
#print("Tuned Training R-squared:", tuned_tree_cv.best_score_.round(4))
```

```
# Tuning Results
# Tuned Parameters  : {'splitter': 'best', 'min_samples_leaf': 11, 'max_depth': 10, 'criterion': 'poisson'}
# Tuned Training R-squared: 0.6783

# New Prediction using tuning parameters result

## Linear Regression
model_name = 'Pruned Regression Tree (Tuned)'

# INSTANTIATING a model object 
model = DecisionTreeRegressor(min_samples_leaf = 11,
                              max_depth        = 10,
                              random_state     = 702,
                              splitter         = 'best',
                              criterion        = 'poisson')
                                
# FITTING to the training data
model_fit = model.fit(x_train, y_train)

# PREDICTING on new data
model_pred = model.predict(x_test)

# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    
# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)
```
Linear Regression
```
## OLS

# naming the model
model_name = 'Linear Regression' # name your model

# model type
model = sklearn.linear_model.LinearRegression()
```
```
# FITTING to the training data
model_fit = model.fit(x_train, y_train)


# PREDICTING on new data
model_pred = model.predict(x_test)


# SCORING the results
model_train_score = model.score(x_train, y_train).round(4)
model_test_score  = model.score(x_test, y_test).round(4)
model_gap         = abs(model_train_score - model_test_score).round(4)
    

# dynamically printing results
model_summary =  f"""\
Model Name:     {model_name}
Train_Score:    {model_train_score}
Test_Score:     {model_test_score}
Train-Test Gap: {model_gap}
"""

print(model_summary)
```

## Part VI: Preparing Submission File for Kaggle
```
# x-data
x_data_kaggle = kaggle_data[x_features].copy()


# y-data
y_data_kaggle = kaggle_data[y_variable]


# Fitting model from above to the Kaggle test data
kaggle_predictions = model.predict(x_data_kaggle)
```
Creating the Kaggle File
```
## Kaggle Submission File ##

# organizing predictions
model_predictions = {"RENTALS" : kaggle_predictions}


# converting predictions into df
model_pred_df = pd.DataFrame(data  = model_predictions,
                             index = df_test.index)
```
```
#!######################!#
#!# name the .csv file #!#
#!######################!#

# name your model
model_pred_df.to_csv(path_or_buf = "./model_output/LASSO.csv",
                     index       = True,
                     index_label = 'ID')
```
