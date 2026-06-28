## Local Generation & Provider Routing

### Basic Backends

- **Hugging Face Transformers (`hf:` prefix)**  
  Example: `--model hf:meta-llama/Llama-3.1-8B-Instruct` (internally runs `transformers.pipeline("text-generation", device_map="auto")`).  
  Pros: simple single-GPU experiments, wide model coverage.  
  Cons: slower throughput than vLLM for large batches, higher memory pressure.  
  Install: `pip install transformers accelerate`

- **vLLM (`vllm:` prefix)**  
  Example: `--model vllm:meta-llama/Llama-3.1-8B-Instruct` (uses `vllm.LLM` + `SamplingParams`).  
  Pros: high-throughput, memory efficient for big datasets.  
  Cons: requires vLLM setup, limited built-in chat templates.  
  Install: `pip install vllm`

- **Provider APIs (default)**  
  Pass `openai/...`, `anthropic/...`, etc. and LiteLLM handles routing via your API keys. No local weights required.

`dementor-generate basic` uses whichever backend you specify. The disguise step (`dementor-disguise`) defaults to LiteLLM; to force a local vLLM endpoint, provide:

```
--openai-api-base http://localhost:8000/v1
--openai-api-key EMPTY
--model openai/meta-llama/Llama-3.1-8B-Instruct  # ensures LiteLLM routes to your server
```

### Example: Running Disguise via Local vLLM

1. **Start vLLM server**
   ```bash
   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --dtype auto --max-model-len 4096
   ```
2. **Generate base outputs (LiteLLM routed to vLLM)**
   ```bash
   dementor-generate \
     --prompts-file data/datasets/chatbot_arena/chatbot_arena_prompts.csv \
     --output-csv data/model-responses/chatbot_arena/full/llama31_8b.csv \
     basic \
     --model openai/meta-llama/Llama-3.1-8B-Instruct \
     --openai-api-base http://localhost:8000/v1 \
     --openai-api-key EMPTY
   ```
3. **Run disguise**
   ```bash
   dementor-disguise \
     --model openai/meta-llama/Llama-3.1-8B-Instruct \
     --disguise-as openai/gpt-4.1-mini \
     --method contrastive \
     --num-samples 200 \
     --openai-api-base http://localhost:8000/v1 \
     --openai-api-key EMPTY
   ```
4. **Score results**
   ```bash
   python -m dementor.scorer pairwise \
     --input data/results/chatbot_arena/disguised/my_run.csv \
     --output data/results/chatbot_arena/scores/my_run_scored/scored.csv
   ```

Chain the steps above — vLLM server launch, then `dementor-generate`, `dementor-disguise`, and `dementor.scorer` — to drive the entire local loop.
