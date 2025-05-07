# Clustering
## 1. Stylistic feature based clustering
Running `bash run.sh` the following clusters all responses (outputs to csv under `clusters`) and samples 1 prompt-response pair from each cluster (outputs to json under `/samples`).

To process a single file:
```bash
python clustering.py --method kmodes --n_clusters 5 --n_samples -1 --data ../../model-responses/gpt-4o_responses.csv
```
## 2. Vibe scores based clustering