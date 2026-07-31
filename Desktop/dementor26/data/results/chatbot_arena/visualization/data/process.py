import pandas as pd

# Filter out specific data from wandb export csv file
# source_file = "./wandb_export_2025-05-16T02_54_03.211-07_00.csv"

# df = pd.read_csv(source_file)

# filtered_df = df[df['method'] == 'stylistic_clustering']

# filtered_df.to_csv("./stylistic_clustering_20250516.csv", index=False)

# Combine dfs from wandb
df1 = pd.read_csv("./random_5_examples_20250724.csv")
df2 = pd.read_csv("./stylistic_clustering_20250724.csv")
df3 = pd.read_csv("./embedding_clustering_20250724.csv")

combined_df = pd.concat([df1, df2, df3])
columns = [
    "Name", 
    "disguise_as", "method", "model",
    "ALL CAPS", "Blockquotes", "Bullets", "Code Formatting", "Emojis", "Exclamations",
    "First-Person Pronouns", "Greeting (Start)", "Header/Title", "Links", "List", "Long Sentences",
    "Markdown", "Math Symbols", "Numbered Steps", "Parentheses", "Questions", "Repetition",
    "Sign-off (End)", "Starts with List", "average_disguised_response_token_length",
    "average_distance_disguise_target", "average_distance_source_target",
    "diff_avg_embedding_distance", "heuristic_avg_score", "heuristic_avg_score_source_target",
    "heuristic_diff", "heuristic_diff_normalized", "length_diff", 
    "source_target_ALL CAPS", "source_target_Blockquotes", "source_target_Bullets", "source_target_Code Formatting",
    "source_target_Emojis", "source_target_Exclamations", "source_target_First-Person Pronouns",
    "source_target_Greeting (Start)", "source_target_Header/Title", "source_target_Links",
    "source_target_List", "source_target_Long Sentences", "source_target_Markdown",
    "source_target_Math Symbols", "source_target_Numbered Steps", "source_target_Parentheses",
    "source_target_Questions", "source_target_Repetition", "source_target_Sign-off (End)",
    "source_target_Starts with List"
]

combined_df = combined_df[columns]

# Add metrics (Normalized Improvement Ratio)
aspects = [
    "ALL CAPS", "Blockquotes", "Bullets", "Code Formatting", "Emojis", "Exclamations",
    "First-Person Pronouns", "Greeting (Start)", "Header/Title", "Links", "List", "Long Sentences", 
    "Markdown", "Math Symbols", "Numbered Steps", "Parentheses", "Questions", "Repetition", 
    "Sign-off (End)", "Starts with List"
]

epsilon = 1e-8  # to prevent division by 0

# Compute per-aspect relative improvement
improvements = []
for aspect in aspects:
    disguised = combined_df[aspect]
    baseline = combined_df[f"source_target_{aspect}"]
    relative_improvement = (disguised - baseline) / (baseline + epsilon)
    improvements.append(relative_improvement)

# Stack and average
nir_matrix = pd.concat(improvements, axis=1)
nir_matrix.columns = aspects
combined_df["heuristic_diff_max"] = nir_matrix.max(axis=1)
combined_df["max_heuristic_diff_name"] = nir_matrix.idxmax(axis=1)

combined_df.to_csv("./combined_3_methods.csv", index=False)



