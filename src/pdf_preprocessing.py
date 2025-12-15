import os
import fitz
import base64
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import MIME_MAPPING, pdf_dir

def process_pdfs(pdf_dir=pdf_dir):
    text_documents = []
    image_chunks = []

    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_dir, filename)
            
            # Text processing step
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=[
                    "\n\n",
                    "\nFigure",
                    "\nTable",
                    "\nAlgorithm",
                    "\n• ",
                    "(?<!\d)\.(?!\d)",
                    "\n",
                    " "
                ]
            )
            chunked_docs = text_splitter.split_documents(docs)
            text_documents.extend(chunked_docs)

            # Image processing step
            pdf_doc = fitz.open(file_path)
            for page_index in range(len(pdf_doc)):
                page = pdf_doc[page_index]
                images = page.get_images(full=True)
                for img in images:
                    xref = img[0]
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    base64_str = base64.b64encode(image_bytes).decode('utf-8')
                    image_format = base_image.get("ext", "png").lower()
                    mime_type = MIME_MAPPING.get(image_format, 'application/octet-stream')
                    image_chunks.append({
                        "file": filename,
                        "page": page_index,
                        "base64_str": base64_str,
                        "mime_type": mime_type
                    })
    return text_documents, image_chunks
