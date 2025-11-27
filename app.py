import config  # loads HF token
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse  # <--- Import this
from schemas import PromptRequest, EmbedRequest
from llm.llm_service import generate_llm_response
from embeddings.embed_service import get_embedding


app = FastAPI(title="GenAI API", version="1.0", root_path="/genaiapi")


# -----------------------------
# TEXT GENERATION ENDPOINT
# -----------------------------
@app.post("/generate/")
async def generate_text(request: PromptRequest):
    try:
        result = generate_llm_response(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Check if the service returned a safety error
        if "error" in result:
            # Return 400 Bad Request (or 403 Forbidden) explicitly
            return JSONResponse(status_code=400, content=result)

        return result

    except Exception as e:
        # This catches unexpected server crashes, not safety violations
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# EMBEDDING ENDPOINT
# -----------------------------
@app.post("/embed/")
async def embed_text(req: EmbedRequest):
    try:
        emb = get_embedding(req.text)

        return {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "embedding_dimension": len(emb),
            "embedding": emb
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))