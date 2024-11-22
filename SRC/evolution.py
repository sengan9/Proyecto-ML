#librerias

import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import yaml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay, precision_recall_curve

#abrir archivo
datos_train = pd.read_csv('Train Data\train.csv')

#entrenamiento

X = datos_train[['Share pledge ratio of controlling shareholders',
        'Pledge ratio of unlimited shares',
        'audit opinion ',
        'Downgrade or negative',
        'Ratio of other receivables to total assets',
        'ROA',
        'Asset liability ratio',
        'Pledge ratio of limited sale shares',
        'ROE',
        'Enterprise age']]
y = datos_train['IsDefault']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = []

#5. XGBoost
model_5 = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_5.fit(X_train, y_train)  
y_pred_xgb = model_5.predict(X_test)
results.append(evaluate_model(y_test, y_pred_xgb, "XGBoost"))