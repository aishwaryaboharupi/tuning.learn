import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10]
}

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred)

# Print results
print("\nBest Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)

# Save results to CSV
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv("gridsearch_results.csv", index=False)
