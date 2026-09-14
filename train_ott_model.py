import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

print("--> Ingesting OTT Churn Dataset...")
df = pd.read_csv('ott_churn_model_dataset.csv')

# Drop rows where target is missing
df = df.dropna(subset=['churn'])
df['churn'] = df['churn'].astype(int)

# Handle missing values
df['gender'] = df['gender'].fillna('Unknown')
df['maximum_days_inactive'] = df['maximum_days_inactive'].fillna(df['maximum_days_inactive'].median())

# Drop non-predictive identifiers
drop_cols = ['year', 'customer_id', 'phone_no']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Binary string mapping
df['multi_screen'] = df['multi_screen'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
df['mail_subscribed'] = df['mail_subscribed'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

# Feature engineering
df['daily_avg_mins'] = df['weekly_mins_watched'] / 7.0
df['mins_per_video'] = df['weekly_mins_watched'] / (df['videos_watched'] + 1)

# Categorical dummies (gender)
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

X = df.drop(columns=['churn'])
y = df['churn']

feature_columns = list(X.columns)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train calibrated Random Forest Classifier
print("--> Training Random Forest Model for OTT Subscriber Churn...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\nModel Performance Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(model, 'ott_rf_model.pkl')
joblib.dump(scaler, 'ott_scaler.pkl')
joblib.dump(feature_columns, 'ott_columns.pkl')

print("\n--> Artifacts successfully exported:")
print(" - ott_rf_model.pkl")
print(" - ott_scaler.pkl")
print(" - ott_columns.pkl")
