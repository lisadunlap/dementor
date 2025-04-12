# dementor
Stealing the souls of LLMs

## Setup
```bash
pip install -r requirements.txt
```

## Running Local Models with VLLM
run `vllm serve` with the huggingface model name and the number of GPUs you want to use

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --dtype half --tensor_parallel_size 4
```

if you want to run multiple models at once, set `--port` to a different port for each model

To test that your model is running, run `python utils_llm.py` (make sure the model name is uncommented in the `test_get_llm_output` function)

## Getting Differences Between Models

```bash
python get_differences.py --input_file <path to file with prompts and model outputs> --batch_size <number of prompts to process at once> --num_rounds <number of rounds to run> --num_final_vibes <number of final vibes to return> --models <model1> <model2>
```

This expects your input file to have a "prompt" column, and two columns with the model outputs (e.g. "friendly_model" and "cold_model").

For example,
```bash
python get_differences.py --input_file data/friendly_and_cold_sample.csv --batch_size 10 --num_rounds 2 --num_final_vibes 5 --models friendly_model cold_model
```

This will run 2 rounds of getting differences between the friendly and cold models, and then reduce the differences to 5 final vibes.
