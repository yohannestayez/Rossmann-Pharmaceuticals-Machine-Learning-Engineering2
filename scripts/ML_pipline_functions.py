import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import logging
import warnings
import pickle
from datetime import datetime

def load_data(train_path, test_path, store_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)
    
    # Merge store information
    train = train.merge(store, on='Store', how='left')
    test = test.merge(store, on='Store', how='left')
    logging.info("Datasets loaded successfully")
    return train, test

def clean_data(train, test):
    # Clean StateHoliday column
    train['StateHoliday'].replace({'0': 0}, inplace=True)
    test['StateHoliday'].replace({'0': 0}, inplace=True)
    
    # Remove rows where store is closed
    reduced_train_df = train[train.Open == 1].copy()
    
    # Convert Date column to datetime
    reduced_train_df['Date'] = pd.to_datetime(reduced_train_df['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    
    # Create year, month, day columns
    for df in [reduced_train_df, test]:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        
    return reduced_train_df, test

def create_pipeline(input_cols, num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    return Pipeline(steps=[('preprocessor', preprocessor)])

def rmspe(y_true, y_pred):
    percentage_error = (y_true - y_pred) / y_true
    percentage_error[y_true == 0] = 0
    squared_percentage_error = percentage_error ** 2
    mean_squared_percentage_error = np.mean(squared_percentage_error)
    return np.sqrt(mean_squared_percentage_error)

def try_model(model, train_inputs, train_targets, val_inputs, val_targets):
    model.fit(train_inputs, train_targets)
    train_preds = model.predict(train_inputs)
    val_preds = model.predict(val_inputs)

    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)

    train_rmspe = rmspe(train_targets, train_preds)
    val_rmspe = rmspe(val_targets, val_preds)

    print(f"Train RMSE: {train_rmse}")
    print(f"Val RMSE: {val_rmse}")
    print(f"Train RMSPE: {train_rmspe}")
    print(f"Val RMSPE: {val_rmspe}")
