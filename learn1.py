# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# 1️⃣ Load Sample Dataset (Handwritten Digits)
digits = load_digits()
X, y = digits.data, digits.target

# 2️⃣ Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Create a Simple Model with Default Parameters
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)  # Train the model
y_pred = model.predict(X_test)  # Make predictions

# 4️⃣ Print the Model’s Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")  # Example output: 0.9456
