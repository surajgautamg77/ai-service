import config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SAFETY_MODEL = "meta-llama/Llama-Guard-3-8B"

# Load Tokenizer and Model
guard_tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL)
guard_model = AutoModelForCausalLM.from_pretrained(
    SAFETY_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Ensure pad token is set (Llama 3 often defaults pad to None)
if guard_tokenizer.pad_token is None:
    guard_tokenizer.pad_token = guard_tokenizer.eos_token

def run_safety_check(text: str):
    """
    Formats the text into a chat structure so Llama Guard knows 
    it is performing a classification task, not text completion.
    """
    messages = [
        {"role": "user", "content": text}
    ]

    # Apply the chat template (essential for Llama 3 based models)
    input_ids = guard_tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(guard_model.device)

    with torch.no_grad():
        output_ids = guard_model.generate(
            input_ids,
            max_new_tokens=20,  # Llama Guard output is very short (e.g., "unsafe\nS1")
            pad_token_id=guard_tokenizer.pad_token_id
        )

    # Decode ONLY the generated part, excluding the input prompt
    # calculate length of input so we can slice the output
    prompt_len = input_ids.shape[-1]
    generated_tokens = output_ids[0][prompt_len:]
    
    result = guard_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return result

def is_unsafe(text: str) -> bool:
    """
    Llama Guard outputs 'safe' or 'unsafe\nS<Code>'.
    We check if 'unsafe' is present in the output string.
    """
    txt = text.lower()
    return "unsafe" in txt