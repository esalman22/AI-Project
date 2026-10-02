import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Generate Synthetic Data (based on notebook analysis)
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Hemoglobin': np.random.uniform(6, 16, n_samples),
    'RDW': np.random.uniform(10, 20, n_samples),
    'MCV': np.random.uniform(70, 110, n_samples),
    'Age': np.random.randint(1, 90, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Pregnancy_Status': np.random.choice(['Yes', 'No', 'Not Applicable'], n_samples),
    'Medical_History': np.random.choice(['None', 'Diabetes', 'Hypertension'], n_samples),
    'Activity_Level': np.random.choice(['Low', 'Moderate', 'High'], n_samples),
    'Tea_Coffee_Intake': np.random.choice(['Low', 'Moderate', 'High'], n_samples)
})

# Feature Engineering (mimicking notebook)
data['Hb_Level'] = pd.cut(data['Hemoglobin'], bins=[0, 11, 13, 20], labels=['Low', 'Normal', 'High'])
data['Cell_Status'] = pd.cut(data['MCV'], bins=[0, 80, 100, 150], labels=['Microcytic', 'Normocytic', 'Macrocytic'])

# Target Variable (Simplified logic for synthetic data)
def classify_anemia(row):
    if row['Hemoglobin'] < 11:
        return 'Iron Deficiency Anemia'
    elif row['Hemoglobin'] < 13 and row['MCV'] < 80:
        return 'Microcytic Anemia'
    else:
        return 'No Anemia'

data['Anemia_Type'] = data.apply(classify_anemia, axis=1)

# Preprocessing
cat_cols = ['Gender', 'Pregnancy_Status', 'Medical_History', 'Activity_Level', 'Tea_Coffee_Intake', 'Hb_Level', 'Cell_Status']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Target encoding
target_le = LabelEncoder()
data['Anemia_Type'] = target_le.fit_transform(data['Anemia_Type'])

# Selected features from notebook: ['Hemoglobin', 'RDW', 'Medical_History', 'Hb_Level', 'Cell_Status']
features = ['Hemoglobin', 'RDW', 'Medical_History', 'Hb_Level', 'Cell_Status']
X = data[features]
y = data['Anemia_Type']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Save Model and Encoders
joblib.dump(rf, 'anemia_rf_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(target_le, 'target_encoder.pkl')
joblib.dump(features, 'features.pkl')

print("Model and assets saved successfully.")
