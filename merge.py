# merge.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE      = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER   = "./anime_model"
SAVE_PATH = "./anime_model_merged"

tokenizer = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, ADAPTER)

print("Merging...")
model = model.merge_and_unload()
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Saved to {SAVE_PATH}")