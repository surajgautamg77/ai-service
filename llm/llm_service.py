from vllm import LLM, SamplingParams
from utils import build_prompt
from safety.safety import run_safety_check, is_unsafe

# Load LLaMA-3 using vLLM
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_memory_utilization=0.5, 
    max_model_len=4096  # Limit context length to save VRAM (Llama 3 supports 8k, but that eats memory)
)


def generate_llm_response(prompt: str, system_prompt: str, max_tokens: int, temperature: float, top_p: float):
    
    # Pre-safety check
    safety_in = run_safety_check(prompt)
    if is_unsafe(safety_in):
        return {"error": "User input violates safety policies.", "safety": safety_in}

    # Build Llama prompt
    full_prompt = build_prompt(prompt, system_prompt)

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Generate output
    outputs = llm.generate([full_prompt], params)
    generated_text = outputs[0].outputs[0].text.strip()

    # Post-safety check
    safety_out = run_safety_check(generated_text)
    if is_unsafe(safety_out):
        return {"error": "Generated output violates safety policies.", "safety": safety_out}

    return {
        "generated_text": generated_text,
        "input_safety": safety_in,
        "output_safety": safety_out
    }
