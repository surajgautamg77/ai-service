# Project Structure

```
```

# File Contents
---
File: safety/safety.py
---

```py
import config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SAFETY_MODEL = "meta-llama/Llama-Guard-3-8B"

guard_tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL)
guard_model = AutoModelForCausalLM.from_pretrained(
    SAFETY_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

def run_safety_check(text: str):
    inputs = guard_tokenizer(text, return_tensors="pt").to(guard_model.device)

    with torch.no_grad():
        output = guard_model.generate(
            **inputs,
            max_new_tokens=60
        )

    result = guard_tokenizer.decode(output[0], skip_special_tokens=True)
    return result

def is_unsafe(text: str) -> bool:
    txt = text.lower()
    return ("unsafe" in txt) or ("disallowed" in txt)


```

---
File: exports.md
---

```md
#!/bin/bash

# --- Configuration ---
OUTPUT_FILE="whole_project_structure.md"

# 1. Exclude specific directories/files ONLY from the project root
ROOT_EXCLUDE_ARRAY=( 
    ".git" 
    ".vscode" 
    "$OUTPUT_FILE" # Always exclude the output file itself
    "export.md"
    "venv"
    ".env"
    "package-lock.json"
)

# 2. NEW: Exclude any directory that matches these names, ANYWHERE in the project
PATTERN_EXCLUDE_ARRAY=(
    "node_modules"
    "dist"
    "build"
    "coverage"
    "__pycache__" # Example for Python projects
)

# 3. Exclude files with these extensions from being read
BINARY_EXTENSIONS=( "png" "jpg" "jpeg" "gif" "ico" "svg" "webp" "woff" "woff2" "ttf" "eot" "otf" "pdf" "zip" "gz" "tar" "rar" "exe" "dll" "so" "a" "lib" "jar" "mp3")

# --- Dynamically build exclusion patterns ---

# For 'tree', both root and pattern exclusions work the same way
COMBINED_TREE_EXCLUDES=("${ROOT_EXCLUDE_ARRAY[@]}" "${PATTERN_EXCLUDE_ARRAY[@]}")
TREE_EXCLUDE_PATTERN=""
for item in "${COMBINED_TREE_EXCLUDES[@]}"; do
    TREE_EXCLUDE_PATTERN+="$item|"
done
TREE_EXCLUDE_PATTERN=${TREE_EXCLUDE_PATTERN%|}

# For 'find', we build the two types of rules separately
ROOT_FIND_ARGS=()
for item in "${ROOT_EXCLUDE_ARRAY[@]}"; do
    ROOT_FIND_ARGS+=(-o -path "./$item")
done

PATTERN_FIND_ARGS=()
for item in "${PATTERN_EXCLUDE_ARRAY[@]}"; do
    # Use -name to match the directory name anywhere
    PATTERN_FIND_ARGS+=(-o -name "$item")
done

# Combine all directory exclusion rules for 'find'
COMBINED_FIND_ARGS=("${ROOT_FIND_ARGS[@]}" "${PATTERN_FIND_ARGS[@]}")
COMBINED_FIND_ARGS=("${COMBINED_FIND_ARGS[@]:1}") # Remove the initial '-o'

# Build binary exclusion rules as before
BINARY_EXCLUDE_ARGS=()
for ext in "${BINARY_EXTENSIONS[@]}"; do
    BINARY_EXCLUDE_ARGS+=(-o -iname "*.$ext")
done
BINARY_EXCLUDE_ARGS=("${BINARY_EXCLUDE_ARGS[@]:1}")

# --- Script ---

# 1. Create the file header and the clean directory tree
{
    echo "# Project Structure"
    echo ""
    echo "\`\`\`"
    # Use the native Linux 'tree' command with the --prune flag for clarity
    tree -aF --prune -I "$TREE_EXCLUDE_PATTERN"
    echo "\`\`\`"
    echo ""
    echo "# File Contents"
} > "$OUTPUT_FILE"

# 2. Find, filter, and append file contents
# Use the new combined rules to prune directories, then filter out binaries
find . \( "${COMBINED_FIND_ARGS[@]}" \) -prune -o -type f -not \( "${BINARY_EXCLUDE_ARGS[@]}" \) -print | while IFS= read -r file; do
    relativePath=$(echo "$file" | sed 's|^\./||')
    extension="${relativePath##*.}"
    if [[ "$relativePath" == "$extension" ]]; then extension="text"; fi
    
    # Append (>>) the content for each file to the existing file
    {
        echo "---"
        echo "File: $relativePath"
        echo "---"
        echo ""
        echo "\`\`\`$extension"
        # Use 'sed' to remove Windows carriage return ('\r') characters for LLM compatibility
        sed 's/\r$//' "$file"
        echo ""
        echo "\`\`\`"
        echo ""
    } >> "$OUTPUT_FILE"
done

echo "✅ Project successfully exported to '$OUTPUT_FILE' (patterns and binaries ignored)."
```

---
File: app.py
---

```py
import config  # loads HF token
from fastapi import FastAPI, HTTPException
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

        return result

    except Exception as e:
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

```

---
File: schemas.py
---

```py
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9
    system_prompt: str | None = None


class EmbedRequest(BaseModel):
    text: str

```

---
File: embeddings/embed_service.py
---

```py
import torch
from transformers import AutoTokenizer, AutoModel

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"

embed_tokenizer = AutoTokenizer.from_pretrained(NOMIC_MODEL, trust_remote_code=True)
embed_model = AutoModel.from_pretrained(NOMIC_MODEL, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model.to(device)
embed_model.eval()


def get_embedding(text: str):
    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=8192
    ).to(device)

    with torch.no_grad():
        outputs = embed_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)[0]
    return embedding.cpu().tolist()

```

---
File: llm/llm_service.py
---

```py
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

```

---
File: config.py
---

```py
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not found in environment.")

login(token=HF_TOKEN)

```

---
File: requirements.txt
---

```txt
accelerate==1.11.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.2
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.11.0
astor==0.8.1
async-timeout==5.0.1
attrs==25.4.0
blake3==1.0.8
cachetools==6.2.1
cbor2==5.7.1
certifi==2025.10.5
cffi==2.0.0
charset-normalizer==3.4.4
click==8.2.1
cloudpickle==3.1.2
compressed-tensors==0.11.0
cupy-cuda12x==13.6.0
depyf==0.19.0
dill==0.4.0
diskcache==5.6.3
distro==1.9.0
dnspython==2.8.0
einops==0.8.1
email-validator==2.3.0
exceptiongroup==1.3.0
fastapi==0.121.1
fastapi-cli==0.0.16
fastapi-cloud-cli==0.3.1
fastrlock==0.8.3
filelock==3.19.1
frozendict==2.4.6
frozenlist==1.8.0
fsspec==2025.9.0
gguf==0.17.1
h11==0.16.0
hf-xet==1.2.0
httpcore==1.0.9
httptools==0.7.1
httpx==0.28.1
huggingface-hub==0.36.0
idna==3.11
interegular==0.3.3
Jinja2==3.1.6
jiter==0.12.0
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
lark==1.2.2
llguidance==0.7.30
llvmlite==0.44.0
lm-format-enforcer==0.11.3
markdown-it-py==4.0.0
MarkupSafe==2.1.5
mdurl==0.1.2
mistral_common==1.8.5
mpmath==1.3.0
msgpack==1.1.2
msgspec==0.19.0
multidict==6.7.0
networkx==3.3
ninja==1.13.0
numba==0.61.2
numpy==2.1.2
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
openai==2.7.2
openai-harmony==0.0.8
opencv-python-headless==4.12.0.88
outlines_core==0.2.11
packaging==25.0
partial-json-parser==0.2.1.1.post6
pillow==11.3.0
prometheus-fastapi-instrumentator==7.1.0
prometheus_client==0.23.1
propcache==0.4.1
protobuf==6.33.0
psutil==7.1.3
py-cpuinfo==9.0.0
pybase64==1.4.2
pycountry==24.6.1
pycparser==2.23
pydantic==2.12.4
pydantic-extra-types==2.10.6
pydantic_core==2.41.5
Pygments==2.19.2
python-dotenv==1.2.1
python-json-logger==4.0.0
python-multipart==0.0.20
PyYAML==6.0.3
pyzmq==27.1.0
ray==2.51.1
referencing==0.37.0
regex==2025.11.3
requests==2.32.5
rich==14.2.0
rich-toolkit==0.15.1
rignore==0.7.6
rpds-py==0.28.0
safetensors==0.6.2
scipy==1.15.3
sentencepiece==0.2.1
sentry-sdk==2.44.0
setproctitle==1.3.7
shellingham==1.5.4
sniffio==1.3.1
soundfile==0.13.1
soxr==1.0.0
starlette==0.49.3
sympy==1.14.0
tiktoken==0.12.0
tokenizers==0.22.1
tomli==2.3.0
torch==2.8.0
torchaudio==2.8.0
torchvision==0.23.0
tqdm==4.67.1
transformers==4.57.1
triton==3.4.0
typer==0.20.0
typer-slim==0.20.0
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.5.0
uvicorn==0.38.0
uvloop==0.22.1
vllm==0.11.0
watchfiles==1.1.1
websockets==15.0.1
xformers==0.0.32.post1
xgrammar==0.1.25
yarl==1.22.0

```

---
File: .gitignore
---

```gitignore
venv

.env
```

---
File: utils.py
---

```py
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

```

