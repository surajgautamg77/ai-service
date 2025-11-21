from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9
    system_prompt: str | None = None


class EmbedRequest(BaseModel):
    text: str
