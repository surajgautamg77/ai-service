from pydantic import BaseModel, Field
from typing import Literal

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    
    # "search_query" = Use this for your short/messy user questions
    # "search_document" = Use this when saving clean text to your DB
    task_type: Literal["search_document", "search_query"] = "search_query"