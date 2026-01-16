conda activate litellm
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=0 python scripts/disguise.py   --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv   --model vllm:meta-llama/Meta-Llama-3.1-8B-Instruct   --disguise_as openai/gpt-4.1-mini   --source_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv   --target_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv   --method contrastive   --num_samples 200


conda activate litellm
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=1 python scripts/disguise.py   --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv   --model vllm:meta-llama/Meta-Llama-3.1-8B-Instruct   --disguise_as openai/gpt-4.1-mini   --source_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv   --target_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv   --method behavioral_based   --num_samples 200



conda activate litellm
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=2 python scripts/disguise.py   --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv   --model vllm:meta-llama/Meta-Llama-3.1-8B-Instruct   --disguise_as openai/gpt-4.1-mini   --source_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv   --target_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv   --method stylistic   --num_samples 200



conda activate litellm
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=3 python scripts/disguise.py   --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv   --model vllm:meta-llama/Meta-Llama-3.1-8B-Instruct   --disguise_as openai/gpt-4.1-mini   --source_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv   --target_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv   --method random_sampling   --num_samples 200

# python scripts/disguise.py   --prompts_file data/datasets/mmlu_pro/mmlu_pro_prompts_eval_200_seed42.csv   --model vllm:meta-llama/Meta-Llama-3.1-8B-Instruct   --disguise_as openai/gpt-4.1-mini   --source_responses data/model-responses/mmlu_pro/full/meta-llama_Meta-Llama-3.1-8B-Instruct_responses.csv   --target_responses data/model-responses/mmlu_pro/full/openai_gpt-4.1-mini_responses.csv   --method embedding_clustering   --num_samples 200

