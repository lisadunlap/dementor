import pandas as pd

df = pd.read_csv("disguising/model-responses/base_500_all_models_2/google_gemma-3-27b-it.csv")
df['prompt'].to_csv("data/gemma_prompts.txt", index=False, header=False)