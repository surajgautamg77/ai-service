from vllm import LLM, SamplingParams
from utils import build_prompt

class LLMService:
    def __init__(self, safety_service):
        print("Loading vLLM Engine...")
        self.safety_service = safety_service
        self.llm = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            gpu_memory_utilization=0.5,
            max_model_len=4096,
            trust_remote_code=True
        )

    def generate(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float, top_p: float):
        # 1. Pre-safety check
        if self.safety_service.is_unsafe(prompt):
            return {"error": "User input violates safety policies."}

        # 2. Build Prompt
        full_prompt = build_prompt(prompt, system_prompt)

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # 3. Generate
        outputs = self.llm.generate([full_prompt], params)
        generated_text = outputs[0].outputs[0].text.strip()

        # 4. Post-safety check
        if self.safety_service.is_unsafe(generated_text):
            return {"error": "Generated output violates safety policies."}

        return {
            "generated_text": generated_text,
            "finish_reason": outputs[0].outputs[0].finish_reason
        }