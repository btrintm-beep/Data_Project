import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

data_frame = pd.read_csv("cardio_clean.csv")

features = data_frame.drop(columns=["cardio"])
labels = data_frame["cardio"]

features_train, features_test, labes_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

knn_scaler =  StanderdScaler()
features_train_scaled = knn_scaler.fit_transform(features_train)
features_test_scaled = knn_scaler.transform(features_test)