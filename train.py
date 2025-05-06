# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

# --- 1. Load and clean training data ---
train_df = pd.read_csv("train.csv")
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.dropna(inplace=True)

# --- 2. Prepare features and labels ---
X = train_df.drop(columns=["id", "will_buy_on_return_visit"], errors='ignore')
y = train_df["will_buy_on_return_visit"]

# --- 3. Encode categorical features ---
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# --- 4. Train/test split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# --- 5. Train Random Forest model ---
model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# --- 6. Evaluate on validation data ---
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))

# --- 7. Load and process test data ---
test_df = pd.read_csv("test.csv")

# Fill missing values
fill_defaults = {
    'avg_time_per_page': test_df['avg_time_per_page'].mean(),
    'medium': test_df['medium'].mode()[0],
    'operatingSystem': test_df['operatingSystem'].mode()[0],
    'city': test_df['city'].mode()[0]
}
for col, val in fill_defaults.items():
    if col in test_df.columns:
        test_df[col].fillna(val, inplace=True)

# Encode test data using training encoders
for col in test_df.select_dtypes(include=["object"]).columns:
    if col in label_encoders:
        le = label_encoders[col]
        test_df[col] = test_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Drop unnecessary columns and predict
X_test = test_df.drop(columns=["id", "unique_session_id"], errors="ignore")
test_preds = model.predict(X_test)

# --- 8. Output results ---
output = test_df[["id"]].copy()
output["will_buy_on_return_visit"] = test_preds
output.to_csv("fix_predicted_data_random_forest.csv", index=False)

print("Prediction completed. Sample result:")
print(output.head())
