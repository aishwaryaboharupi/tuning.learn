import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Optuna results
df = pd.read_csv("optuna_results.csv")

# Set up visualization style
sns.set(style="whitegrid")

# Plot accuracy over trials
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x=df.index, y="value", marker="o", color="b")
plt.title("Optuna Hyperparameter Tuning Progress")
plt.xlabel("Trial Number")
plt.ylabel("Accuracy")
plt.show()
