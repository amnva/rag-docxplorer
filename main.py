import faiss
import fitz
import torch
import pickle
from openai import OpenAI
from src.config import pdf_dir, device
from src.pdf_preprocessing import process_pdfs
from src.embeddings import generate_embeddings, DimensionalAdapter
from src.search import hybrid_search
from src.answer_generator import generate_answer
from sentence_transformers import SentenceTransformer

#openai_api_key = "API key"

if __name__ == "__main__":
    # Initialize client with API key
    client = OpenAI(api_key=openai_api_key)

    # Process docs and generate embeddings
    text_docs, image_chunks = process_pdfs(pdf_dir)
    embeddings, metadata, adapter = generate_embeddings(text_docs, image_chunks, device)

    # Create and save index
    index = faiss.IndexFlatIP(512)
    index.add(embeddings)
    faiss.write_index(index, "tech_index.faiss")
    torch.save(adapter.state_dict(), "adapter_weights.pth")

    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Initialize Q&A
    while True:
        query = input("\nAsk about the documents (type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break

        results = hybrid_search(
            query=query,
            index=index,
            metadata=metadata,
            text_encoder=text_encoder,
            adapter=adapter,
            top_k=5
        )
        answer = generate_answer(client, query, results)
        
        print(f"\nAnswer:\n{answer}")
        print("\nSources:")
        for res in results:
            source_info = f"{res.source} page {res.page}"
            print(f"- {res.type.capitalize()} from {source_info}")
