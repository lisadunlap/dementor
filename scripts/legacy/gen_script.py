import os  # moved to scripts/legacy
import glob
import math

# Base paths for both methods
base_paths = [
    "../model-responses/disguised/stylistic_clustering",
    "../model-responses/disguised/stylistic_clustering_resample"
]

# Collect all csv files
all_csv_files = []
for base_path in base_paths:
    csv_pattern = os.path.join(base_path, "**/temp0.7/*.csv")
    all_csv_files.extend(glob.glob(csv_pattern, recursive=True))

# Calculate number of files per bash script
files_per_script = math.ceil(len(all_csv_files) / 8)

# Generate 8 bash files
for i in range(8):
    with open(f"run_scoring_{i+1}.sh", "w") as f:
        f.write("#!/bin/bash\n\n")
        
        start_index = i * files_per_script
        end_index = min((i + 1) * files_per_script, len(all_csv_files))
        
        for csv_file in all_csv_files[start_index:end_index]:
            command = f"python disguising/llm_scorer.py --input_file {csv_file.replace('..', 'disguising')}\n"
            f.write(command)
    
    # Make bash script executable
    os.chmod(f"run_scoring_{i+1}.sh", 0o755)

print(f"Generated 8 bash files with {files_per_script} commands each (except possibly the last one)")
