import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load GridSearchCV results
df = pd.read_csv("gridsearch_results.csv")

# Extract only relevant columns
df = df[["param_n_estimators", "param_max_depth", "param_min_samples_split", "mean_test_score"]]

# Rename columns for easier plotting
df.columns = ["n_estimators", "max_depth", "min_samples_split", "accuracy"]

# Set up visualization style
sns.set(style="whitegrid")

# Plot Accuracy vs. n_estimators
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="n_estimators", y="accuracy", hue="max_depth", style="min_samples_split", markers=True, dashes=False)
plt.title("GridSearchCV: Effect of Hyperparameters on Accuracy")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.legend(title="Depth & Min Samples Split")
plt.show()
