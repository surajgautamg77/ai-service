import torch
# Make sure you installed this: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_id: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer on {self.device}...")
        
        # trust_remote_code=True is required for Nomic
        self.model = SentenceTransformer(model_id, trust_remote_code=True, device=self.device)

        self.prompts = {
            "search_query": "search_query: ",
            "search_document": "search_document: "
        }

    def generate(self, text: str, task_type: str = "search_query"):
        prefix = self.prompts.get(task_type, "search_query: ")
        text_with_prefix = prefix + text

        # This handles tokenization, pooling, and normalization automatically
        embedding = self.model.encode(
            text_with_prefix, 
            convert_to_tensor=False,
            normalize_embeddings=True 
        )

        return embedding.tolist()