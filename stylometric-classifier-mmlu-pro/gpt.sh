export WANDB_MODE=disabled


python scripts/disguise.py   --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv   --model openai/gpt-4.1-mini   --disguise_as meta-llama/Meta-Llama-3.1-8B-Instruct   --source_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv   --target_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv   --method stylistic --num_samples 200

python scripts/disguise.py   --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv   --model openai/gpt-4.1-mini   --disguise_as meta-llama/Meta-Llama-3.1-8B-Instruct   --source_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv   --target_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv   --method random_sampling --num_samples 200

# python scripts/disguise.py   --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv   --model openai/gpt-4.1-mini   --disguise_as meta-llama/Meta-Llama-3.1-8B-Instruct   --source_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv   --target_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv   --method embedding_clustering --num_samples 200