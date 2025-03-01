import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Store results for visualization
results = []

# Try different hyperparameter values
for n in [50, 100, 150, 200]:  # Number of trees
    for depth in [10, 20, 30]:  # Tree depth
        for min_split in [2, 5, 10]:  # Min samples to split
            model = RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_split=min_split, random_state=42)
            model.fit(X_train, y_train)  # Train the model
            y_pred = model.predict(X_test)  # Make predictions
            acc = accuracy_score(y_test, y_pred)  # Calculate accuracy

            # Store results
            results.append({"n_estimators": n, "max_depth": depth, "min_samples_split": min_split, "accuracy": acc})

            print(f"n_estimators: {n}, max_depth: {depth}, min_samples_split: {min_split}, Accuracy: {acc:.4f}")

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Print Best Hyperparameters
best_params = df_results.loc[df_results['accuracy'].idxmax()]
print("\nBest Hyperparameters:", best_params.to_dict())

# Save to CSV (optional)
df_results.to_csv("tuning_results.csv", index=False)
