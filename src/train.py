import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from preprocess import load_nsl_kdd, basic_preprocessing, split_dataset

# Load NSL-KDD dataset
df_train, df_test = load_nsl_kdd("../data/KDDTrain+.txt", "../data/KDDTest+.txt")

# Remove the "difficulty" column if present (not needed for training)
if "difficulty" in df_train.columns:
    df_train.drop(columns=["difficulty"], inplace=True)

if "difficulty" in df_test.columns:
    df_test.drop(columns=["difficulty"], inplace=True)

# Preprocess data (encode categorical columns)
df_train = basic_preprocessing(df_train)
df_test = basic_preprocessing(df_test)

# Check for missing labels in training set that exist in test set
train_labels = set(df_train["label"].unique())
test_labels = set(df_test["label"].unique())

missing_labels = test_labels - train_labels
if missing_labels:
    print(f"⚠️ Warning: These labels exist in df_test but not in df_train: {missing_labels}")

    # Add missing labels to df_train with dummy values to prevent errors
    for missing_label in missing_labels:
        fake_row = pd.DataFrame([df_train.iloc[0]])  # Copy an existing row
        fake_row["label"] = missing_label  # Replace label with missing one
        df_train = pd.concat([df_train, fake_row], ignore_index=True)

# Ensure the LabelEncoder learns all possible labels
all_labels = sorted(list(train_labels | test_labels))  # Union of known and unknown labels

# Initialize LabelEncoder with all possible labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)  # Learn all labels

# Convert labels to numbers
df_train["label"] = label_encoder.transform(df_train["label"])
df_test["label"] = label_encoder.transform(df_test["label"])

# Split dataset into features (X) and labels (y)
X_train, X_test, y_train, y_test = split_dataset(df_train)

# Check class distribution BEFORE applying SMOTE
unique_classes, class_counts = np.unique(y_train, return_counts=True)
print("📊 Class distribution before SMOTE:")
for cls, count in zip(unique_classes, class_counts):
    print(f"Class {cls}: {count} samples")

# Identify rare classes (less than 2 samples)
min_samples_before_smote = 2  # SMOTE requires at least k_neighbors + 1
valid_classes = [cls for cls, count in zip(unique_classes, class_counts) if count >= min_samples_before_smote]

# Filter out classes that are too rare for SMOTE
mask = np.isin(y_train, valid_classes)
X_train_filtered = X_train[mask]
y_train_filtered = y_train[mask]

# Apply SMOTE with k_neighbors=1 to avoid the error
smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_filtered, y_train_filtered)

# Check class distribution AFTER SMOTE
unique_classes_res, class_counts_res = np.unique(y_train_resampled, return_counts=True)
print("📊 Class distribution after SMOTE:")
for cls, count in zip(unique_classes_res, class_counts_res):
    print(f"Class {cls}: {count} samples")

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Create the "models" directory if it does not exist
os.makedirs("../models", exist_ok=True)

# Save trained model
joblib.dump(model, "../models/random_forest.joblib")

print("✅ Model trained and saved successfully!")

# Test the model on a real attack sample
df_attacks = df_test[df_test["label"] != label_encoder.transform(["normal"])[0]]  # Filter attack samples

if not df_attacks.empty:
    X_attack = df_attacks.iloc[0, :-1].values.reshape(1, -1)

    # Verify feature count before making a prediction
    print(f"📊 Features in X_attack before prediction: {X_attack.shape[1]}")
    print(f"📊 Expected features by the model: {model.n_features_in_}")

    # Ensure X_attack has the correct number of features
    if X_attack.shape[1] > model.n_features_in_:
        X_attack = X_attack[:, :model.n_features_in_]

    y_attack_real = label_encoder.inverse_transform([df_attacks.iloc[0, -1]])[0]  # Decode actual attack class

    # Predict with the model
    y_pred = model.predict(X_attack)[0]
    y_proba = model.predict_proba(X_attack)[0]

    print(f"✅ Actual class: {y_attack_real}")
    print(f"🔎 Model prediction: {label_encoder.inverse_transform([y_pred])[0]}")
    print(f"📊 Class probabilities: {y_proba}")
else:
    print("⚠️ No attacks found in the test dataset.")
