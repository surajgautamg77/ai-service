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
