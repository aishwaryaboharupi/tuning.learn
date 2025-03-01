import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results from CSV
df = pd.read_csv("tuning_results.csv")

# Set up visualization style
sns.set(style="whitegrid")

# Plot Accuracy vs. n_estimators
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="n_estimators", y="accuracy", hue="max_depth", style="min_samples_split", markers=True, dashes=False)
plt.title("Effect of Hyperparameters on Accuracy")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.legend(title="Depth & Min Split")
plt.show()
