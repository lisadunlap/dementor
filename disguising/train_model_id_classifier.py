from unsloth import FastLanguageModel, FastModel
import torch
import pandas as pd
from datasets import Dataset
import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import numpy as np
import nltk
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets import concatenate_datasets, Dataset
import random
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import train_on_responses_only
import wandb
import os

from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

from vllm import LLM, SamplingParams


name = "model_id"
run = wandb.init(project="model-id-classifier", name=name)
# Update wandb config with our config dictionary

results_folder = os.path.join("results", name.replace(" ", "_"))
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

model_name = "llama-3.1-8b-4b"
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True
eval_only = True

##############################
# Load model
##############################
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    full_finetuning = False, # [NEW!] We have full finetuning now!
    load_in_8bit = False 
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                                        # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj",
                      "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,                              # Supports any, but = 0 is optimized
    bias = "none",                                 # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth",        # True, or [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    random_state = 3407,
    use_rslora = False,                            # We support rank stabilized LoRA
    loftq_config = None,                           # And LoftQ
)


##############################
# Data
##############################
random.seed(42)
responses = []
models = ['llama-3-8b-instruct_responses.csv', 'phi-4-multimodal-instruct_responses.csv', 'gpt-4o_responses.csv', 'gpt-3.5_responses.csv', 'gpt-4o-mini_responses.csv', 'qwen-7b_responses.csv', 'molmo-7b-o_responses.csv']
for f in models:
  df = pd.read_csv(f"disguising/model-responses/{f}").drop_duplicates(subset=["prompt"])
  df["prompt"] = df["prompt"].str.strip('"')
  df["model_response"] = df["model_response"].str.strip()
  df["model"] = f.split(".")[0].replace("_responses", "")
  responses.append(df)

num_responses = len(responses)
responses = pd.concat(responses)
train_prompts = responses.prompt.unique()
# filter any prompts that have less than 10 responses
unique_prompts = responses.groupby('prompt').size()[responses.groupby('prompt').size() == num_responses].index.tolist()
train_prompts = unique_prompts[:6000]
test_prompts = [p for p in unique_prompts if p not in train_prompts] 
train_responses = responses[responses.prompt.isin(train_prompts)]
test_responses = responses[responses.prompt.isin(test_prompts)]

prompt_format = """Below you will be given a prompt and a response from an LLM. Your task is to determine which of the following models is most likely to have generated the response.

Here are the possible models:
- {models}

Here is the prompt and response:

## Prompt:
{prompt}

## Model response:
{response}

Please respond with the model name only, no other text.
"""

possible_models = "\n- ".join(list(responses.model.unique()))

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def turn_into_conversations(entry):
    systems_prompt = "You are a helpful assistant."
    user_prompt = prompt_format.format(prompt = entry["prompt"], response = entry["model_response"], models = possible_models)
    # Return the messages structure directly
    conversations = {"messages": [{"content": systems_prompt, "role": "system"}, {"content": user_prompt, "role": "user"}, {"content": entry["model"], "role": "assistant"}]}
    return conversations


db_data_processed = [turn_into_conversations(entry) for entry in train_responses.to_dict(orient="records")]
# Apply formatting to create the dataset for training
dataset = Dataset.from_list(db_data_processed)
dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"Formatted dataset size for training: {len(dataset)}")

print(f"Total dataset size for training: {len(dataset)}")
# print first 10 rows of dataset
print(dataset[0])
dataset = dataset.shuffle(seed = 3407)


##############################
# Eval
##############################
if eval_only:
    llm = LLM(model=model, max_model_len=2048)

    sampling_params = SamplingParams(
        max_model_len=2048,
    )

    prompts = [turn_into_conversations(entry) for entry in test_responses.to_dict(orient="records")][:10]
    results = llm.generate(prompts, sampling_params)
    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Result: {result.text}")
        print("\n")

##############################
# Training
##############################

# Define the PredictionLogger Callback
class PredictionLogger(TrainerCallback):
    """
    Callback to log model predictions on random samples during training.
    """
    def __init__(self, tokenizer, dataset, num_samples=10, log_every_n_steps=20):
        self.tokenizer = tokenizer
        self.dataset = dataset # The dataset *before* formatting_prompts_func
        self.num_samples = num_samples
        self.log_every_n_steps = log_every_n_steps
        # Initialize wandb table here or lazily in on_step_end if preferred
        self.wandb_table = None

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        # Log every N steps, and not on the first step (step 0)
        if state.global_step > 0 and state.global_step % self.log_every_n_steps == 0:
            if self.wandb_table is None:
                 self.wandb_table = wandb.Table(columns=["step", "prompt", "ground_truth", "prediction"])

            # Ensure model is available
            if model is None:
                print("Warning: Model not provided to PredictionLogger callback.")
                return

            # Set model to evaluation mode
            was_training = model.training
            model.eval()

            try:
                # Sample data
                indices = random.sample(range(len(self.dataset)), self.num_samples)
                samples = self.dataset.select(indices)

                prompts = []
                ground_truths = []
                predictions = []

                for sample in samples:
                    messages = sample['messages']
                    # Prepare prompt (all messages except the last assistant message)
                    prompt_messages = messages[:-1] # Assumes last message is assistant's response
                    # Ensure add_generation_prompt=True for models that need it
                    prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt_text)

                    # Get ground truth
                    ground_truth = messages[-1]['content']
                    ground_truths.append(ground_truth)

                    # Generate prediction
                    inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048 - 512).to(model.device) # Leave space for generation
                    # Use generate method of the model
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512, # Limit generated tokens
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=True, # Optional: use sampling
                        top_p=0.9,      # Optional: nucleus sampling
                        temperature=0.7 # Optional: temperature
                    )
                    # Decode only the generated part, skipping special tokens
                    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                    prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    predictions.append(prediction)

                # Add data to WandB Table
                for prompt, gt, pred in zip(prompts, ground_truths, predictions):
                    # Add data row by row
                    self.wandb_table.add_data(state.global_step, prompt, gt, pred)

            except Exception as e:
                print(f"Error during prediction logging at step {state.global_step}: {e}")
            finally:
                # Log the table if it has new rows
                if self.wandb_table and len(self.wandb_table.data) > 0:
                     # Log the entire table at once
                     wandb.log({"predictions_samples": self.wandb_table})
                     # Reset table for the next logging interval to avoid logging old data repeatedly
                     self.wandb_table = wandb.Table(columns=["step", "prompt", "ground_truth", "prediction"])

                # Set model back to original mode (train or eval)
                if was_training:
                    model.train()

# Instantiate the callback using the unformatted dataset for sampling
prediction_logger = PredictionLogger(
    tokenizer=tokenizer,
    dataset=dataset, # Use the dataset with 'messages' structure
    log_every_n_steps=20 # Log every 20 steps
)

# this is for phi, I tried mistral but it seems to break easily
print("Training on responses only")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    callbacks = [prediction_logger],
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 20,
        num_train_epochs = 1,
        learning_rate = 0.0001,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", 
        max_grad_norm = 1.0,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer_stats = trainer.train()
#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save the model after training
model.save_pretrained(model_name) # Local saving
tokenizer.save_pretrained(model_name)

FastLanguageModel.for_inference(model)

# db_data_processed = [turn_into_conversations(entry) for entry in train_responses.to_dict(orient="records")]
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
model.push_to_hub_merged(model_name, tokenizer, save_method = "merged_16bit", token = "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")