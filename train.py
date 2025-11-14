import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'C:\Users\USER\PycharmProjects\ML Reinforcement\Data\hypertension_dataset.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

le = LabelEncoder()
categorical_cols = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data['Has_Hypertension'] = data['Has_Hypertension'].map({'Yes':1, 'No':0})

x = data.drop('Has_Hypertension', axis=1)
y = data['Has_Hypertension']

x_train, x_temp,y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

print(f'Train: {x_train.shape}, Validation: {x_val.shape}, Test: {x_test.shape}')

smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
print(y_train_res.value_counts())

scaler = StandardScaler()
x_train_res = scaler.fit_transform(x_train_res)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

from xgboost import XGBClassifier

xgb = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(
    x_train_res, y_train_res,
    eval_set=[(x_val, y_val)],
    verbose=True
)

y_pred = xgb.predict(x_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

import joblib
import os

model_dir = r'C:\Users\USER\PycharmProjects\ML Reinforcement\Model'
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'trained_model.pkl')
scaler_path = os.path.join(model_dir, 'fixed_scaler.pkl')

joblib.dump(xgb, model_path)
joblib.dump(scaler, scaler_path)


