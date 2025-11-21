def build_prompt(user_prompt: str, system_prompt: str | None = None):
    if system_prompt is None:
        system_prompt = (
            "You are an AI assistant. Respond clearly, accurately and politely."
        )

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
