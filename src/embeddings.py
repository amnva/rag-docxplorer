import torch
import clip
import numpy as np
import base64
import faiss
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from src.config import device
from src.models import TextChunk, ImageChunk

class DimensionalAdapter(torch.nn.Module):
    def __init__(self, input_dim=384, output_dim=512):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

def generate_embeddings(text_docs, image_chunks, device):
    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    adapter = DimensionalAdapter().to(device)

    embeddings = []
    metadata = []

    for doc in text_docs: # text processing
        content = doc.page_content
        with torch.no_grad():
            text_embed = text_encoder.encode(content, convert_to_tensor=True)
            projected = adapter(text_embed.to(device)).cpu().numpy()
        embeddings.append(projected.reshape(1, -1))
        metadata.append(TextChunk(
            type="text",
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", 0),
            content=content
        ))

    for img in image_chunks: # image processing
        image_data = base64.b64decode(img["base64_str"])
        image = Image.open(BytesIO(image_data))
        
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
            embed = outputs[0].cpu().numpy()
        
        embeddings.append(embed.reshape(1, -1))
        metadata.append(ImageChunk(
            type="image",
            source=img["file"],
            page=img["page"],
            content=img["base64_str"],
            mime_type=img["mime_type"]
        ))

    embeddings = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    return embeddings, metadata, adapter
