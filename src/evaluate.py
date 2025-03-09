import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from preprocess import load_nsl_kdd, basic_preprocessing

# -----------------------------------------------------------------------------
# 1. Load and preprocess the dataset
# -----------------------------------------------------------------------------
# Load NSL-KDD dataset (both train and test are loaded to obtain all labels)
df_train, df_test = load_nsl_kdd("../data/KDDTrain+.txt", "../data/KDDTest+.txt")

# Remove "difficulty" column if present (not needed for evaluation)
if "difficulty" in df_train.columns:
    df_train.drop(columns=["difficulty"], inplace=True)
if "difficulty" in df_test.columns:
    df_test.drop(columns=["difficulty"], inplace=True)

# Preprocess both datasets (this encodes categorical columns, etc.)
df_train = basic_preprocessing(df_train)
df_test = basic_preprocessing(df_test)

# -----------------------------------------------------------------------------
# 2. Create a LabelEncoder using the union of train and test labels
# -----------------------------------------------------------------------------
all_labels = sorted(list(set(df_train["label"].unique()) | set(df_test["label"].unique())))
le = LabelEncoder()
le.fit(all_labels)

# Convert the labels in the test set using the learned encoder
y_true = le.transform(df_test["label"])

# -----------------------------------------------------------------------------
# 3. Prepare the test features and ensure consistency with training
# -----------------------------------------------------------------------------
# Drop the label column and any "difficulty" column, so that X contains only the features used in training.
X = df_test.drop(columns=["label", "difficulty"], errors="ignore")
print(f"Test set feature columns: {list(X.columns)}")

# Load the pre-trained Random Forest model
model = joblib.load("../models/random_forest.joblib")

# -----------------------------------------------------------------------------
# 4. Make predictions and compute evaluation metrics
# -----------------------------------------------------------------------------
y_pred = model.predict(X)

# Compute and display the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
print("Confusion Matrix:")
print(cm)

# For classification_report, use only the classes present in the test set.
unique_y_true = np.unique(y_true)
target_names = le.inverse_transform(unique_y_true)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=unique_y_true, target_names=target_names))

# -----------------------------------------------------------------------------
# 5. Compute and plot the ROC Curve (binary: "normal" vs "attack")
# -----------------------------------------------------------------------------
# For a binary ROC, consider "normal" as the negative class and everything else as "attack".
y_binary_true = np.where(df_test["label"] == "normal", 0, 1)

# Get predicted probabilities; assume the probability for "normal" is at the index corresponding to "normal"
normal_index = list(le.classes_).index("normal")
y_pred_proba = model.predict_proba(X)[:, normal_index]
# Define attack probability as 1 - probability of normal
attack_proba = 1 - y_pred_proba

fpr, tpr, thresholds = roc_curve(y_binary_true, attack_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Normal vs Attack)")
plt.legend(loc="lower right")
plt.show()
