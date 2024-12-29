# Import Libraries
import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import category_encoders as ce
from category_encoders import TargetEncoder
import numpy as np
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


# Function to deal with Data Wrangling:
def handle_missing_values(df):
  """Handles missing values in a DataFrame."""

  for column in df.columns:
    if df[column].dtype == 'object':
      # For categorical columns, fill with the mode (most frequent value)
      df.loc[:, column] = df[column].fillna(df[column].mode()[0])
    else:
      # For numerical columns, check for skewness
      if df[column].isnull().any():
        if df[column].isnull().sum() > len(df) / 2:  # More than half of values are null
          df = df.drop(column, axis=1)  # Drop the column
        elif abs(df[column].skew()) < 0.8:  # Use absolute value of skewness
          df.loc[:, column] = df[column].fillna(df[column].mean())
        else:
          # For skewed data, consider using median instead of mean
          df.loc[:, column] = df[column].fillna(df[column].median())
  return df


# encode Label:
# For train:
def label_encode_train(data, labelcols):
    label_encoders = {}
    for col in labelcols:
        label_encoders[col] = LabelEncoder()
        label_encoders[col].fit(data[col])
        data[col] = label_encoders[col].transform(data[col])
    return data, label_encoders

# For test:
def label_encode_test(data, labelcols, label_encoders):
    for col in labelcols:
        data[col] = label_encoders[col].transform(data[col])
    return data


# encode: Target 
# For train:
def target_encode_train(data, target_cols, target):
    # Dictionary to store the fitted encoders
    targ_encoders = {}
    for col in target_cols:  
        target_encoder = ce.TargetEncoder(cols=[col])
        # Fit the encoder on the training data and transform it
        data[col] = target_encoder.fit_transform(data[col], data[target])
        # Store the fitted encoder for future use on the test set
        targ_encoders[col] = target_encoder
    return data, targ_encoders

# For test: 
def target_encode_test(data, targ_encoders, target_cols):
    for col in target_cols:
        if col not in targ_encoders:
            raise KeyError(f"Encoder for column '{col}' not found in targ_encoders.")
        target_encoder = targ_encoders[col]
        # Handle unseen categories by using the encoder's default behavior or assigning a fallback value
        if data[col].isnull().sum() > 0:
            print(f"Missing values detected in '{col}' before encoding.")
        
        # Transform the test data using the fitted encoder
        try:
            data[col] = target_encoder.transform(data[col])
        except Exception as e:
            print(f"Error transforming '{col}' in test data: {e}")
            raise e
    return data


# for reducing the df's to not have kernal crashes:
def red_mem_usage(df):
  """reducing memory usage to avoid kernal crashes when modeling"""
  start_df_memory_usage = df.memory_usage().sum() / 1024 ** 2
  print('Memory usage of this df is: ', start_df_memory_usage, 'MB')
  Null_list = [] # keeping track of cols that had nulls before handling them
  
  for col in df.columns:
    if df[col].dtype != object: # only having numerical values
      print('--------------------------------') 
      print('column: ', col)
      print('data type before: ', df[col].dtype)
      
      is_int = False 
      max = df[col].max()
      min = df[col].min()
      
      # seeing if cols can be changed to int64
      as_int = df[col].astype(np.int64)
      result = (df[col]- as_int).sum()      
      if result > -0.01 and result < 0.01:
        is_int = True
      
      if is_int:
        if min >= 0:
          if max < 255:
            df.loc[:, col] = df[col].astype(np.uint8)
          elif max < 65535:
            df.loc[:, col] = df[col].astype(np.uint16)
          elif max < 4294967295:
            df.loc[:, col] = df[col].astype(np.uint32) 
          else:
            df.loc[:, col] = df[col].astype(np.uint64)
        else:
          if min > np.iinfo(np.int8).min and max < np.iinfo(np.int8).max:
            df.loc[:, col] = df[col].astype(np.int8)
          elif min > np.iinfo(np.int16).min and max < np.iinfo(np.int16).max:
            df.loc[:, col] = df[col].astype(np.int16)     
          elif min > np.iinfo(np.int32).min and max < np.iinfo(np.int32).max:
            df.loc[:, col] = df[col].astype(np.int32)
          elif min > np.iinfo(np.int64).min and max < np.iinfo(np.int64).max:
            df.loc[:, col] = df[col].astype(np.int64)  
      else:
        df.loc[:, col] = df[col].astype(np.float32)
      
      print('data type after: ', df[col].dtype) # displaying the new data type
      print('--------------------------------')   
      
  print('___ memory usage after: __') # final result
  memory_usage = df.memory_usage().sum() / 1024 ** 2
  print("Memory usage is:", memory_usage, "MB")

  return df, Null_list
      