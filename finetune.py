import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig

MODEL     = "microsoft/Phi-3-mini-4k-instruct"
DATA_FILE = "training_data.jsonl"
SAVE_PATH = "./anime_model"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
))
model.print_trainable_parameters()

data = [json.loads(l) for l in open(DATA_FILE) if l.strip()]

def tokenize(batch):
    results = {"input_ids": [], "attention_mask": [], "labels": []}
    for instruction, output in zip(batch["instruction"], batch["output"]):
        prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
        full   = prompt + output + "<|end|>"
        prompt_ids = tokenizer(prompt, truncation=True, max_length=1024)["input_ids"]
        full_enc   = tokenizer(full,   truncation=True, max_length=1024, padding="max_length")
        labels = full_enc["input_ids"].copy()
        # Mask the prompt tokens — only train on the answer
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
        results["input_ids"].append(full_enc["input_ids"])
        results["attention_mask"].append(full_enc["attention_mask"])
        results["labels"].append(labels)
    return results

dataset = Dataset.from_list(data).map(tokenize, batched=True, remove_columns=["instruction", "input", "output"])
dataset = dataset.map(lambda x: {"labels": x["input_ids"]})

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="./tmp",
        fp16=True,
        report_to="none",
    ),
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Training...\n")
trainer.train()

print("\nSaving...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Done. Model saved to {SAVE_PATH}")