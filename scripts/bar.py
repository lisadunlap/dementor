import matplotlib.pyplot as plt
import pandas as pd

# Data from wandb runs
data = {
    "heuristic_match_mean": {
        "baseline": 0.34861,
        "random_sampling": 0.3493,
        "vibe_based": 0.34953,
        "contrastive_with_al_examples": 0.34965,
    },
    "num_samples_scored": {
        "baseline": 502,
        "random_sampling": 1002,
        "vibe_based": 1502,
        "contrastive_with_al_examples": 2002,
    },
    "semantic_score_mean": {
        "baseline": 3.97211,
        "random_sampling": 3.9491,
        "vibe_based": 3.9494,
        "contrastive_with_al_examples": 3.94955,
    },
    "stylistic_score_mean": {
        "baseline": 2.69522,
        "random_sampling": 2.74651,
        "vibe_based": 2.747,
        "contrastive_with_al_examples": 2.72328,
    },
}

# Convert to DataFrame
df = pd.DataFrame(data)

# --- Combined bar chart ---
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(kind="bar", ax=ax)
plt.title("Comparison of Runs (wandb summaries)")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

# --- Separate bar charts for each metric ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, metric in zip(axes.flatten(), df.columns):
    df[metric].plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_title(metric)
    ax.set_ylabel("Value")
    ax.set_xticklabels(df.index, rotation=45)

plt.suptitle("Comparison of Runs by Individual Metrics", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
