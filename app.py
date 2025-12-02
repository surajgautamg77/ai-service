import config
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from schemas import PromptRequest, EmbedRequest
from embeddings.embed_service import EmbeddingService
from safety.safety import SafetyService
from llm.llm_service import LLMService

# Global Model Holders
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load models on startup
        # Note: Ensure you have enough VRAM for all three!
        models["safety"] = SafetyService()
        models["llm"] = LLMService(safety_service=models["safety"])
        models["embed"] = EmbeddingService()
        
        print("✅ All models loaded successfully.")
        yield
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        raise e
    finally:
        models.clear()

app = FastAPI(title="GenAI API", version="1.0", root_path="/genaiapi", lifespan=lifespan)

# -----------------------------
# TEXT GENERATION ENDPOINT
# -----------------------------
@app.post("/generate/")
def generate_text(request: PromptRequest):
    llm_svc = models.get("llm")
    if not llm_svc:
        raise HTTPException(status_code=503, detail="LLM service not initialized")

    try:
        result = llm_svc.generate(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        if "error" in result:
            return JSONResponse(status_code=400, content=result)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# EMBEDDING ENDPOINT
# -----------------------------
@app.post("/embed/")
def embed_text(req: EmbedRequest):
    embed_svc = models.get("embed")
    if not embed_svc:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")

    try:
        # Generate embedding (Fixed at 768 dims)
        emb = embed_svc.generate(
            text=req.text,
            task_type=req.task_type
        )

        return {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "task_type": req.task_type,
            "embedding_dimension": 768, 
            "embedding": emb
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))