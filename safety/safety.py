import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SafetyService:
    def __init__(self, model_id: str = "meta-llama/Llama-Guard-3-8B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Safety Model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=self.device 
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def is_unsafe(self, text: str) -> bool:
        """
        Returns True if the content is classified as unsafe.
        """
        messages = [{"role": "user", "content": text}]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=20,
                pad_token_id=self.tokenizer.pad_token_id
            )

        prompt_len = input_ids.shape[-1]
        generated_tokens = output_ids[0][prompt_len:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
        
        return "unsafe" in result