# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 23:12:46 2023

@author: Group 4
"""

'''
Imports
'''
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel, chi2
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.tree import DecisionTreeClassifier


# Set data file path
filename = 'KSI.csv'
path = r'D:\Study\Semester 4\Supervised\Project'
fullpath = os.path.join(path, filename)

# Load the data
df = pd.read_csv(fullpath)
pd.set_option('display.max_columns', None)
df.info()
df.describe()
df.isna().sum()
df.corr()

# The exploration with plots is done in PowerBI

# Drop unnecessary columns that have a lot of null values
columns_to_drop = ['FATAL_NO', 'INJURY','INITDIR','VEHTYPE','MANOEUVER','PEDTYPE','PEDACT', 
                   'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND', 'X', 'Y', 'INDEX_', 'ACCNUM', 
                   'WARDNUM', 'DIVISION', 'ObjectId', 'OFFSET', 'INVAGE']

df.drop(columns_to_drop, axis=1, inplace=True)

# Replace 'Yes' with 1 and fill missing values with 0 for specified columns
columns_to_replace = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH',
                      'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']

df[columns_to_replace] = df[columns_to_replace].replace('Yes', 1).fillna(0)

# Checking unique values
df['ACCLOC'].unique()
df['DRIVCOND'].unique()
df['DRIVACT'].unique()
df['IMPACTYPE'].unique()
df['ACCLASS'].unique()
df['TRAFFCTL'].unique()
df['LOCCOORD'].unique()

# Transforming drivcond
def transform_drivcond(data):
    replace_drivcond = {
        'Normal': 0,
        'Ability Impaired, Alcohol Over .08': 1,
        'Inattentive': 1,
        'Fatigue': 1,
        'Other': 1,
        'Had Been Drinking': 1,
        'Ability Impaired, Drugs': 1,
        'Medical or Physical Disability': 1,
        'Ability Impaired, Alcohol': 1,
        'Unknown': np.nan
    }
    return data.replace(replace_drivcond)

def transform_drivact(data):
    replace_drivact = {
        'Driving Properly': 1,
        'Exceeding Speed Limit': 0,
        'Disobeyed Traffic Control': 0,
        'Following too Close': 0,
        'Lost control': 0,
        'Failed to Yield Right of Way': 0,
        'Improper Passing': 0,
        'Improper Turn': 0,
        'Other': 0,
        'Speed too Fast For Condition': 0,
        'Improper Lane Change': 0,
        'Wrong Way on One Way Road': 0,
        'Speed too Slow': 0,
    }
    return data.replace(replace_drivact)

def transform_acclass(data):
    replace_acclass = {
        'Non-Fatal Injury': 0,
        'Fatal': 1,
        'Property Damage Only': 0
    }
    return data.replace(replace_acclass)

def transform_invtype(data):
    driver_types = ['Driver', 'Truck Driver', 'Motorcycle Driver', 'Moped Driver']
    pedestrian_types = ['Pedestrian', 'Wheelchair', 'Pedestrian - Not Hit']
    cyclist_types = ['Cyclist', 'In-Line Skater', 'Cyclist Passenger']
    
    if data in driver_types:
        return 'Driver'
    elif data in pedestrian_types:
        return 'Pedestrian'
    elif data in cyclist_types:
        return 'Cyclist'
    else:
        return 'Other'
  
def transform_road_class(data):
    replace_roadclass = {
        'Minor Arterial': 'Arterial',
        'Collector': 'Collector',
        'Major Arterial': 'Arterial',
        'Local': 'Local',
        'Expressway': 'Expressway',
        'Expressway Ramp': 'Expressway',
        'Major Arterial Ramp': 'Arterial',
        'Other': 'Other',
        'Pending': 'Other',
        'Laneway': 'Other',
        'unknown': 'Other'
    }
    return data.replace(replace_roadclass)

def transform_traffctl(data):
    replace_traffctl = {
        'No Control': 0,
        'Stop Sign': 1,
        'Yield Sign': 1,
        'School Guard': 1,
        'Traffic Gate': 1,
        'Police Control': 1,
        'Streetcar (Stop for)': 1,
        'Traffic Signal': 1,
        'Pedestrian Crossover': 1,
        'Traffic Controller': 1
    }
    return data.replace(replace_traffctl)

def transform_loccoord(data):
    replace_loccoord = {
        'Exit Ramp Southbound': 0,
        'Mid-Block (Abnormal)': 0,
        'Intersection': 1,
        'Mid-Block': 0,
        'Park, Private Property, Public Lane': 0,
        'Exit Ramp Westbound': 1,
        'Entrance Ramp Westbound': 0
    }
    return data.replace(replace_loccoord)

def dayschedule(value):
    value = int(value)  # Convert the time value from string to integer
    if value >= 0 and value < 1200:
        return 'Morning'
    elif value >= 1200 and value <= 1700:
        return 'Afternoon'
    else:
        return 'Night'

# Transform drivcond
df['ABNORMAL'] = transform_drivcond(df['DRIVCOND'])
df.drop(['DRIVCOND'], axis=1, inplace=True)

# Transform drivact
df['PROPER_DRIVING'] = transform_drivact(df['DRIVACT'])
df.drop(['DRIVACT'], axis=1, inplace=True)

# Transform acclass
df.dropna(subset=['ACCLASS'], inplace=True)
df['ACCLASS'] = transform_acclass(df['ACCLASS'])

# Transform invtype
df['INVTYPE'] = df['INVTYPE'].apply(transform_invtype)

# Transform road class
df['ROAD_CLASS'] = transform_road_class(df['ROAD_CLASS'])

# Transform tarffctl
df['TRAFFCTL'] = transform_traffctl(df['TRAFFCTL'])

# Transform loccoord
df['INTERSECTION'] = transform_loccoord(df['LOCCOORD'])
df.drop(['LOCCOORD'], axis=1, inplace=True)

# Split date into month and day columns
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y/%m/%d %H:%M:%S')
df['MONTH'] = df['DATE'].dt.month
df['DAY'] = df['DATE'].dt.day
df.drop('DATE', axis=1, inplace=True)

df['TIME'] = df['TIME'].apply(dayschedule)

# Reset index
df.reset_index(drop=True, inplace=True)

'''
Train-Test Split
'''
X = df.drop(['ACCLASS'], axis=1)
y = df['ACCLASS']

# Import stratified train-test-split
strat = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=76)

# Split the data into train and test sets
for train_index, test_index in strat.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

'''
Pipeline
'''
def create_pre_pipe(df):
    numerical_columns = list(df.select_dtypes(include=np.number).columns)

    category_columns = list(df.select_dtypes(include=object).columns)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, category_columns)
        ], verbose_feature_names_out=False)
    
    return preprocessor

'''
Feaature Importance & Model Training
'''
# Create a pipeline that combines the preprocessor with the RandomForestClassifier
clf = ImbPipeline(steps=
                    [('preprocessor', create_pre_pipe(X_train)),
                        ('smote', SMOTE(random_state=76)),
                        ('variance', VarianceThreshold(0.1)),
                        ('feature_selection', SelectKBest(chi2, k=10)),
                        ('classifier', RandomForestClassifier(random_state=76))
                    ])

# Fit the model
clf.fit(X_train, y_train)

# Get feature importance
importances = clf.named_steps['classifier'].feature_importances_

# Get feature names
feature_names = clf.named_steps['preprocessor'].transformers_[0][2] + \
                list(clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())
    
feature_names = feature_names[:len(importances)]

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort the DataFrame by importance
feature_importances.sort_values(by='importance', ascending=False, inplace=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature importance')
plt.tight_layout()
plt.show()

# # Print the accuracy score
print("Train Accuracy:", accuracy_score(y_train, clf.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

X_train = X_train[feature_importances['feature']]
X_test = X_test[feature_importances['feature']]


'''
PART B
XGBoost
'''
classifiers = {
    'XGBoost': XGBClassifier(random_state=76),
    'SVM': SVC(random_state=76),
    'Decision Tree': DecisionTreeClassifier(random_state=76),
    'Random Forest': RandomForestClassifier(random_state=76),
    'Logistic Regression': LogisticRegression(random_state=76)
}

param_grids = {
    'XGBoost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.1, 0.01, 0.05]
    },
    'SVM': {
        'classifier__kernel': ['linear', 'rbf', 'poly'],
        'classifier__C': [0.01, 0.1, 0.5, 1],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__degree': [2, 3]
    },
    'Decision Tree': {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [5, 10, 15, 20, 25, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'Logistic Regression': {
        'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'classifier__C': [0.01, 0.1, 0.5, 1],
        'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'classifier__max_iter': [100, 200, 300, 400, 500]
    }
}

new_pre_pipe = create_pre_pipe(X_train)

with open("results.txt", "w") as results_file:
    for classifier_name, classifier in classifiers.items():
        results_file.write(f"Training {classifier_name}...\n")
        print(f"Training {classifier_name}...\n")

        param_grid = param_grids[classifier_name]

        pipeline = ImbPipeline(steps=[
            ('preprocessor', new_pre_pipe),
            ('smote', SMOTE(random_state=76)),
            ('classifier', classifier)
        ])

        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   verbose=10,
                                   cv=5)

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

        results_file.write(f"Best parameters {best_params}\n")
        
        train_pred = best_estimator.predict(X_train)
        test_pred = best_estimator.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        results_file.write(f"Best {classifier_name} Train Accuracy: {train_accuracy:.4f}\n")
        results_file.write(f"Best {classifier_name} Test Accuracy: {test_accuracy:.4f}\n")
        
        confusion = confusion_matrix(y_test, test_pred)
        results_file.write("Confusion matrix:\n")
        results_file.write(str(confusion) + "\n")
        
        classification_rep = classification_report(y_test, test_pred)
        results_file.write("Classification Report:\n")
        results_file.write(classification_rep + "\n")
        
        print(f"Best parameters {best_params}")
        print(f"Best {classifier_name} Train Accuracy:", accuracy_score(y_train, train_pred))
        print(f"Best {classifier_name} Test Accuracy:", accuracy_score(y_test, test_pred))
        print("Confusion matrix \n", confusion_matrix(y_test, test_pred))
        print("Classification Report \n", classification_report(y_test, test_pred))
        
        # Cross validation
        scores = cross_val_score(best_estimator, X_train, y_train, cv=5, n_jobs=-1)
        print(f"The cross val scores for training are: {scores}")
        print(f"The mean score of cross_val_score for training is: {scores.mean()}")
        results_file.write(f"The cross val scores for training are: {scores}\n")
        results_file.write(f"The mean score of cross_val_scor for training is: {scores.mean()}\n")
        
        # Save the best estimator
        joblib.dump(best_estimator, path + f'/{classifier_name.lower().replace(" ", "_")}_best_estimator.pkl')
        
        results_file.write("\n")

print("Results saved to results.txt")
