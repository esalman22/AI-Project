import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load Real Data
data = pd.read_csv('IDA_dataset_enriched_fixed.csv')

# 2. Preprocessing
data.loc[data['Gender'] == 'Male', 'Pregnancy_Status'] = 'Not Applicable'
data['Pregnancy_Status'] = data['Pregnancy_Status'].fillna('Not Applicable')
data['Medical_History'] = data['Medical_History'].fillna('None')
data['Tea_Coffee_Intake'] = data['Tea_Coffee_Intake'].fillna('Moderate')

# Feature Engineering
data['Hb_Level'] = pd.cut(data['Hemoglobin'], bins=[0, 11, 13, 20], labels=['Low', 'Normal', 'High'])
data['Cell_Status'] = pd.cut(data['MCV'], bins=[0, 80, 100, 150], labels=['Microcytic', 'Normocytic', 'Macrocytic'])
data['Blood_Score'] = (data['Hemoglobin'] + data['RDW'] + data['MCV']) / 3

# Categorical columns to encode
cat_cols = ['Gender', 'Pregnancy_Status', 'Medical_History', 'Activity_Level', 'Tea_Coffee_Intake', 'Hb_Level', 'Cell_Status']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Target encoding
target_le = LabelEncoder()
data['Anemia_Type'] = target_le.fit_transform(data['Anemia_Type'])

# Use ALL features as requested by the user's test snippet
features = [
    'Hemoglobin', 'RDW', 'MCV', 'Age', 'Gender', 'Pregnancy_Status', 
    'Medical_History', 'Activity_Level', 'Tea_Coffee_Intake', 
    'Hb_Level', 'Cell_Status', 'Blood_Score'
]
X = data[features]
y = data['Anemia_Type']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Save Model and Assets
joblib.dump(rf, 'anemia_rf_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(target_le, 'target_encoder.pkl')
joblib.dump(features, 'features.pkl')

print("Model and assets saved successfully using ALL features.")
print("Features used:", features)
