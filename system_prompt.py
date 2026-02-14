import pandas as pd
from tqdm import tqdm
from utils_llm import get_llm_output 
import concurrent.futures

input_path = "/home/nazcol/dementor/nazcol@cthulhu6.ist.berkeley.edu/disguising/minimodel_responsesnazx1x2x3x4.csv"
output_path = input_path.replace(".csv", "x5.csv")

df = pd.read_csv(input_path)
assert "prompt" in df.columns, "CSV must contain a 'prompt' column."
df = df.sample(n=500, random_state=42)  # Set random_state for reproducibility

# Add a new column 'x1' with the run number
#run_number = 1  

#get the {vibe}
system_prompt = """
Structured and Organized Responses: To solve the equation, first isolate the variable. Then, apply the inverse operations step by step.  
Step-by-Step Instructional Format: 1. Start by gathering your materials. 2. Next, follow the measurements. 3. Finally, combine the ingredients.  
Focus on Providing Relevant Details: When discussing photosynthesis, it's important to include the roles of sunlight, carbon dioxide, and water.  
Concise and Precise Answers: The capital of France is Paris.  
Use of Bullet Points or Numbered Lists: Feature 1: User-friendly interface. - Feature 2: Cross-platform accessibility.
Strong thematic coherence: The responses are centered around a clear main idea, maintaining alignment throughout to ensure a unified message.
Precise content alignment with the prompt: The output stays closely tied to the topic or question, delivering relevant and directly applicable information.
Focused and specific answers: Responses are clear and directly address the user’s query with actionable insights and targeted detail.
Relevant subject matter: The generated content remains within the scope of the user’s request, providing information that is topically appropriate and meaningful.
Fulfillment of user expectations: The model delivers responses that match the expected depth and scope, offering comprehensive and well-developed content.
"""

disguised_responses = []
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(get_llm_output, f"Prompt: {prompt}", model="gpt-4o-mini", system_prompt=system_prompt): prompt for prompt in df["prompt"].tolist()}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Disguising gpt-3.5"):
        try:
            output = future.result()
        except Exception as e:
            print(f"Error: {e}")
            output = "ERROR"
        disguised_responses.append(output)

df["gpt35_disguised_as_4omini"] = disguised_responses
#df["x1"] = run_number
df.to_csv(output_path, index=False)
print(f"\n✅ Saved disguised outputs to: {output_path}")
