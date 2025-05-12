#!/bin/bash

SESSION_NAME="dementor"

# Start a detached screen session
screen -dmS "$SESSION_NAME"

# Create windows, each activating conda and running the script with a different model
MODEL="meta-llama/Llama-3.1-8B-Instruct"
for DISGUISE_AS in "OpenGVLab/InternVL3-9B" "google/gemma-3-4b-it" "gpt-4o-mini" "gpt-4o" "gpt-3.5" "allenai/Molmo-7B-O-0924" "microsoft/Phi-4-multimodal-instruct" "Qwen/Qwen2.5-VL-7B-Instruct"; do
    screen -S "$SESSION_NAME" -X screen bash -c 'conda activate dementor && python disguising/disguise.py --model '"$MODEL"' \
--disguise_as '"$DISGUISE_AS"' --method stylistic_clustering --num_samples 1000 --test; exec bash'
done



OpenGVLab/InternVL3-9B 
google/gemma-3-1b-it 
google/gemma-3-4b-it
gpt-4o-mini 
gpt-4o 
gpt-3.5 
allenai/Molmo-7B-O-0924 
microsoft/Phi-4-multimodal-instruct 
Qwen/Qwen2.5-VL-7B-Instruct

gpu "python disguising/disguise.py --model meta-llama/Meta-Llama-3-8B-Instruct --disguise_as Qwen/Qwen2.5-VL-7B-Instruct --method stylistic_clustering_resample"