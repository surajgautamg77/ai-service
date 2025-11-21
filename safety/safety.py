import config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SAFETY_MODEL = "meta-llama/Llama-Guard-3-8B"

guard_tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL)
guard_model = AutoModelForCausalLM.from_pretrained(
    SAFETY_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

def run_safety_check(text: str):
    inputs = guard_tokenizer(text, return_tensors="pt").to(guard_model.device)

    with torch.no_grad():
        output = guard_model.generate(
            **inputs,
            max_new_tokens=60
        )

    result = guard_tokenizer.decode(output[0], skip_special_tokens=True)
    return result

def is_unsafe(text: str) -> bool:
    txt = text.lower()
    return ("unsafe" in txt) or ("disallowed" in txt)

