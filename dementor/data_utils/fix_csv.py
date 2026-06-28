#!/usr/bin/env python3
"""
Fix the malformed CSV file by properly parsing and re-writing it with correct CSV formatting.
"""
import csv
import pandas as pd
from pathlib import Path

def fix_malformed_csv(input_path: str, output_path: str):
    """Fix a CSV file that has newlines breaking the format."""
    
    print(f"Reading malformed CSV: {input_path}")
    
    # Read the file line by line to reconstruct proper CSV rows
    rows = []
    current_row = []
    in_quoted_field = False
    quote_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    for line_num, line in enumerate(lines):
        line = line.rstrip('\n\r')
        
        if line_num == 0:
            # Header row
            rows.append(['prompt', 'model_response', 'model'])
            continue
            
        # Count quotes in the line
        line_quotes = line.count('"')
        quote_count += line_quotes
        
        # If we're in a quoted field, accumulate the line
        if in_quoted_field or (line.startswith('"') and not line.endswith('"')):
            current_row.append(line)
            in_quoted_field = quote_count % 2 == 1
        else:
            # This should be a complete row
            if current_row:
                # Join accumulated lines and parse as CSV
                full_row = ' '.join(current_row)
                try:
                    # Use csv.reader to properly parse the row
                    import io
                    reader = csv.reader(io.StringIO(full_row))
                    parsed_row = next(reader)
                    if len(parsed_row) == 3:
                        rows.append(parsed_row)
                    else:
                        print(f"Warning: Row {len(rows)} has {len(parsed_row)} fields, skipping")
                except Exception as e:
                    print(f"Warning: Could not parse row {len(rows)}: {e}")
                current_row = []
            
            # Check if this line starts a new row
            if line.count(',') >= 2:  # Has at least 2 commas (3 fields)
                try:
                    import io
                    reader = csv.reader(io.StringIO(line))
                    parsed_row = next(reader)
                    if len(parsed_row) == 3:
                        rows.append(parsed_row)
                    else:
                        current_row = [line]
                        in_quoted_field = True
                except:
                    current_row = [line]
                    in_quoted_field = True
            else:
                current_row = [line]
                in_quoted_field = True
    
    # Handle any remaining accumulated row
    if current_row:
        full_row = ' '.join(current_row)
        try:
            import io
            reader = csv.reader(io.StringIO(full_row))
            parsed_row = next(reader)
            if len(parsed_row) == 3:
                rows.append(parsed_row)
        except Exception as e:
            print(f"Warning: Could not parse final row: {e}")
    
    print(f"Parsed {len(rows)-1} data rows (plus header)")
    
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
        print("Usage: python fix_csv.py <input_csv> <output_csv>")
        sys.exit(1)
    
    fix_malformed_csv(sys.argv[1], sys.argv[2])
