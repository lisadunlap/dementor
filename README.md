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