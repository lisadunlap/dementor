#!/bin/bash

# Set the directory to search
DIRECTORY="../../model-responses"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi

# Loop through all CSV files in the directory
for file_path in $(find "$DIRECTORY" -name "*.csv"); do
    # Extract the filename without path and extension
    filename=$(basename "$file_path" .csv)
    
    echo "Processing file: $filename"
    
    # Run the clustering script for each file
    python clustering.py --method kmodes --n_clusters 5 --n_samples -1 --data "$DIRECTORY/$filename.csv"
    
    # Optional: add a separator between runs for better readability
    echo "----------------------------------------"
done

echo "All files processed successfully!"