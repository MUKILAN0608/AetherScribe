# LLM package
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,      # FAST + LOW VRAM
        torch_dtype=torch.float16
    )

    return model, tokenizer
