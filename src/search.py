import faiss
import torch
import numpy as np
from src.config import device

def hybrid_search(query, index, metadata, text_encoder, adapter, top_k=5):
    with torch.no_grad():
        query_embed = text_encoder.encode(query, convert_to_tensor=True)
        projected = adapter(query_embed.to(device)).cpu().numpy()
    
    query_embed = projected.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embed)
    distances, indices = index.search(query_embed, top_k)
    return [metadata[i] for i in indices[0]]
