import pandas as pd
import re

# Load your CSV
df = pd.read_csv("old_comparison_results.csv")

# Define a function to extract the average total score from the text summary
def extract_avg_score(text):
    # Look for a line starting with "Average Total Score:" followed by a number
    match = re.search(r"Average Total Score:\s*([\d\.]+)", text)
    if match:
        return float(match.group(1))
    return 0.0

# Apply the extraction function to the comparison_results column
df["discrepancy"] = df["comparison_results"].apply(extract_avg_score)

# Sort rows by the discrepancy score in descending order (assuming a higher score favors GPT-4o)
df_sorted = df.sort_values(by="discrepancy")

# Select the top 5 rows with the largest discrepancy
top5_large_discrepancy = df_sorted.head(10)

# Display the key columns
print(top5_large_discrepancy[["prompt", "gpt35_response", "gpt4omini_response", "discrepancy"]])
