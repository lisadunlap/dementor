#!/usr/bin/env python3
"""
Better fix for the malformed CSV by looking for prompt patterns.
"""
import csv
import re
from pathlib import Path

def fix_malformed_csv_v2(input_path: str, output_path: str):
    """Fix CSV by looking for prompt patterns and reconstructing rows."""
    
    print(f"Reading malformed CSV: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by lines and process
    lines = content.split('\n')
    
    # Find all lines that look like prompts (start with a question word or common patterns)
    prompt_patterns = [
        r'^[A-Z][^,]*\?',  # Questions ending with ?
        r'^[A-Z][^,]*\s+[^,]*\?',  # Questions with more text
        r'^[A-Z][^,]*\s+[^,]*\s+[^,]*\?',  # Longer questions
    ]
    
    rows = [['prompt', 'model_response', 'model']]
    current_prompt = None
    current_response = []
    current_model = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Check if this looks like a prompt (starts a new row)
        is_prompt = False
        for pattern in prompt_patterns:
            if re.match(pattern, line):
                is_prompt = True
                break
        
        if is_prompt and ',' in line:
            # This looks like a prompt line with CSV structure
            parts = line.split(',', 2)  # Split into max 3 parts
            if len(parts) >= 3:
                # Save previous row if we have one
                if current_prompt and current_response and current_model:
                    response_text = ' '.join(current_response).strip()
                    rows.append([current_prompt, response_text, current_model])
                
                # Start new row
                current_prompt = parts[0].strip()
                current_response = [parts[1].strip()] if parts[1].strip() else []
                current_model = parts[2].strip() if len(parts) > 2 else 'gpt-4o'
                i += 1
                continue
        
        # If we're in a response, accumulate lines
        if current_prompt:
            # This is part of the response
            current_response.append(line)
        
        i += 1
    
    # Save the last row
    if current_prompt and current_response and current_model:
        response_text = ' '.join(current_response).strip()
        rows.append([current_prompt, response_text, current_model])
    
    print(f"Reconstructed {len(rows)-1} data rows (plus header)")
    
    # Write the fixed CSV
    print(f"Writing fixed CSV: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for row in rows:
            writer.writerow(row)
    
    print(f"Fixed CSV written with {len(rows)-1} responses")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python fix_csv_v2.py <input_csv> <output_csv>")
        sys.exit(1)
    
    fix_malformed_csv_v2(sys.argv[1], sys.argv[2])
