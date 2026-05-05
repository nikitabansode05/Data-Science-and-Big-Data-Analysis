# Import libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

# -------------------------------
# Load Dataset (iris.csv)
# -------------------------------
# If you have local file:
df = pd.read_csv("iris.csv")

# OR use built-in dataset


# -------------------------------
# Features and Target
# -------------------------------
X = df.drop("species", axis=1)
y = df["species"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Naïve Bayes Model
# -------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# -------------------------------
# Convert to binary (for TP, FP, TN, FN calculation)
# We'll treat one class as positive (e.g., class 0 vs rest)
# -------------------------------
# -------------------------------
# Binary conversion (Class 0 vs Rest)
# -------------------------------
y_test_bin = (y_test == 0).astype(int)
y_pred_bin = (y_pred == 0).astype(int)

# Force confusion matrix to be 2x2
cm_binary = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])

print("\nBinary Confusion Matrix:\n", cm_binary)

TN, FP, FN, TP = cm_binary.ravel()

print("\nTN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)

# -------------------------------
# Metrics Calculation
# -------------------------------
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = (FP + FN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

print("\nPerformance Metrics:")
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)