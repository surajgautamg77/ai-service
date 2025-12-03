from vllm import LLM, SamplingParams

class LLMService:
    def __init__(self):
        print("Loading vLLM Engine (Llama 3.1)...")
        self.llm = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_memory_utilization=0.8,
            max_model_len=8192,
            trust_remote_code=True
        )

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        # The prompt is now passed directly as received from the API
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        outputs = self.llm.generate([prompt], params)
        generated_text = outputs[0].outputs[0].text.strip()

        return {
            "generated_text": generated_text,
            "finish_reason": outputs[0].outputs[0].finish_reason
        }