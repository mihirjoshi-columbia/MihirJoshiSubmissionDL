'''

██████╗ ██╗         ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
██╔══██╗██║         ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ 
██║  ██║██║            ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗
██║  ██║██║            ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
██████╔╝███████╗       ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
╚═════╝ ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                                           
@Author: Mihir Joshi
@Date: 2024-11-19
@Version: 1.3

This script trains a Gradient Boosting Regressor model on the given dataset and generates predictions for unseen dog racing data.
Prior to running this script, ensure that the following files are present in the working directory:
1. df.csv: The training dataset
2. unseendf.csv: The unseen dataset

The script uses the following libraries and automatically installs them if not already present:
1. pandas
2. numpy
3. scikit-learn
4. joblib

The script performs the following steps:
1. Load the training and unseen data.
2. Perform date parsing and enhanced feature engineering.
3. Encode categorical features.
4. Define the feature list and prepare the data.
5. Scale the features using StandardScaler.
6. Train a Gradient Boosting Regressor model with reasonable parameters.
7. Evaluate the model using Mean Squared Error on the validation set.
8. Generate predictions for the unseen data.
9. Save the predictions to a CSV file and save the trained model and feature scaler.
'''

# This code automatically installs the required libraries if not already present (just in case machine that is running this code doesn't have them)
import sys
import subprocess
import pkg_resources

def install_requirements():
    required = {
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib'
    }
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    if missing:
        print(f"Installing missing packages: {missing}") # Install missing packages
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        print("All required packages installed successfully!")

try:
    install_requirements()
except Exception as e:
    print(f"Error installing requirements: {e}") # If you encounter any issues, please install the required libraries manually
    sys.exit(1)

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.base import BaseEstimator
import time
from joblib import parallel_backend, Parallel, delayed
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('df.csv')
unseen_df = pd.read_csv('unseendf.csv')
original_stadium = unseen_df['stadium'].copy() # Had some issues with the stadium column, so saving a copy for later

# Date parsing
training_date_format = '%m/%d/%y' # adjust dt formatting
unseen_date_format = '%Y-%m-%d'    

df['date1'] = pd.to_datetime(df['date1'], format=training_date_format)
df['date2'] = pd.to_datetime(df['date2'], format=training_date_format)

unseen_df['date1'] = pd.to_datetime(unseen_df['date1'], format=unseen_date_format)
unseen_df['date2'] = pd.to_datetime(unseen_df['date2'], format=unseen_date_format)

# Enhanced Feature Engineering
# Design Decision:
# - I have created a few advanced features based on the given data to improve the model's performance.
# - These features include age in days
# - Days between races
# - Speed of dogs
# - Interaction features between age
# - Speed and distance
# - Performance ratio
# - Trap performance
# - I have also encoded the categorical feature 'stadium' using one-hot encoding.
# - The feature list includes the advanced features along with the encoded 'stadium' columns.

def create_advanced_features(data, training=True):
    date_format = training_date_format if training else unseen_date_format
    data['age_days'] = (data['date1'] - pd.to_datetime(data['birthdate'], format=date_format)).dt.days
    data['days_between_races'] = (data['date2'] - data['date1']).dt.days
    data['speed1'] = data['distance1'] / data['time1']
    data['avg_time_per_meter1'] = data['time1'] / data['distance1']
    data['is_same_distance'] = (data['distance1'] == data['distance2']).astype(np.int8)  # Use int8 instead of int64
    data['is_same_trap'] = (data['trap1'] == data['trap2']).astype(np.int8)
    data['age_days_between_interaction'] = data['age_days'] * data['days_between_races']
    data['speed_distance_interaction'] = data['speed1'] * data['distance1']
    data['performance_ratio'] = data['time1'] / data['time1'].mean()
    if training:
        data['trap_performance'] = data.groupby('trap1')['time2'].transform('mean')
    else:
        data['trap_performance'] = 0
    return data

# Apply advanced feature engineering
df = create_advanced_features(df) # Training data -> last column is time2
unseen_df = create_advanced_features(unseen_df, training=False)
df = pd.get_dummies(df, columns=['stadium'], drop_first=True) # One-hot encoding because 'stadium' is a categorical feature
unseen_df = pd.get_dummies(unseen_df, columns=['stadium'], drop_first=True) # Ensure unseen_df has the same columns
missing_cols = set(df.columns) - set(unseen_df.columns)
for col in missing_cols:
    unseen_df[col] = 0 # Add missing columns in unseen_df
unseen_df = unseen_df[df.columns.drop('time2')]

# Define features -> Advanced features + stadium columns
features = [
    'age_days', 'days_between_races', 'time1', 'distance1', 'trap1',
    'speed1', 'avg_time_per_meter1', 'is_same_distance', 'is_same_trap',
    'age_days_between_interaction', 'speed_distance_interaction', 
    'performance_ratio', 'trap_performance'
] + [col for col in df.columns if 'stadium' in col]

# Prepare data
X = df[features]
y = df['time2']

# Scale features -> StandardScaler is used to scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simple model with reasonable parameters
print("Training model...")
best_model = GradientBoostingRegressor(
    n_estimators=200, # Number of boosting stages to be run
    learning_rate=0.1, # Step size shrinkage used in update to prevent overfitting (Reason: learning rate of 0.1 means each new tree contributes only 10% of its potential correction preventing overfitting)
    max_depth=4, # Maximum depth of the individual regression estimators
    min_samples_split=2, # Minimum number of samples required to split an internal node
    min_samples_leaf=1, # Minimum number of samples required to be at a leaf node (Reason: 1 because we want to split nodes until pure)
    subsample=0.8, # Fraction of samples used to fit each base learner (Reason: 0.8 means 80% of the data is used to fit each tree)
    random_state=42 # Reason: (Answer to the Ultimate Question of Life, the Universe, and Everything) Big douglass adams fan
)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
n_cores = multiprocessing.cpu_count()
with parallel_backend('threading', n_jobs=n_cores):
    print(f"Training model using {n_cores} cores...")
    best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation Mean Squared Error: {mse:.2f}")
unseen_X = scaler.transform(unseen_df[features])
print("Generating predictions...")
chunk_size = 10000
if len(unseen_X) > chunk_size:
    print("Generating predictions in chunks...")
    predictions = []
    for i in range(0, len(unseen_X), chunk_size):
        chunk = unseen_X[i:i + chunk_size]
        pred_chunk = best_model.predict(chunk)
        predictions.extend(pred_chunk)
    unseen_df['predtime'] = predictions
else:
    unseen_df['predtime'] = best_model.predict(unseen_X)
unseen_df['stadium'] = original_stadium
desired_columns = [ 'stadium', 'birthdate', 'date1', 'time1', 'distance1', 'trap1', 'comment1', 'date2', 'distance2', 'trap2', 'predtime' ]
unseen_df = unseen_df[desired_columns]
print("Saving results...")
unseen_df.to_csv('mypred.csv', index=False, float_format='%.3f')
print("Predictions saved to mypred.csv")
joblib.dump(best_model, 'gb_model.joblib')
joblib.dump(scaler, 'feature_scaler.joblib')
