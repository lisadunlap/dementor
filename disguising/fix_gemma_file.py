import pandas as pd
import re
import csv

def clean_text(text: str):
    """Remove excess quotes, compress multiple quotes, strip whitespace/newlines"""
    if not isinstance(text, str):
        return text
    # Replace multiple quotes with single quote
    text = re.sub(r'"{2,}', '"', text)
    # Strip leading/trailing whitespace and quotes
    text = text.strip().strip('"').strip()
    # Collapse internal newlines
    text = re.sub(r'[\r\n]+', ' ', text)
    return text

def fix_gemma_file():
    # Read the files using absolute paths
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gemma_file = os.path.join(project_root, "disguising", "model-responses", "base_500_all_models_2", "google_gemma-3-27b-it.csv")
    reference_file = os.path.join(project_root, "disguising", "model-responses", "base_500_all_models_2", "OpenGVLab_InternVL3-8B.csv")
    
    # Read the reference file to get the correct format
    reference_df = pd.read_csv(reference_file)
    
    # Read the gemma file
    gemma_df = pd.read_csv(gemma_file)
    
    # Clean textual columns
    for col in ['prompt', 'messages', 'model_response']:
        if col in gemma_df.columns:
            gemma_df[col] = gemma_df[col].apply(clean_text)
    
    # Ensure same columns and order as reference
    missing_cols = [c for c in reference_df.columns if c not in gemma_df.columns]
    for c in missing_cols:
        gemma_df[c] = ""
    gemma_df = gemma_df[reference_df.columns]
    
    # Save the fixed file
    gemma_df.to_csv(gemma_file, index=False)
    print(f"Fixed file saved to: {gemma_file}")

if __name__ == "__main__":
    fix_gemma_file()
