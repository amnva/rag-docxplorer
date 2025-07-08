## 🔍📄🤖 RAG-DocXplorer

**RAG-DocXplorer** is a modular Retrieval-Augmented Generation (RAG) framework for intelligent question answering over scientific documents. It supports PDF preprocessing, semantic chunk retrieval with embeddings and LLM-based answer generation.

## Features

- 🔍 **PDF Preprocessing**: Clean and parse documents for chunk-based embedding
- 📚 **Embedding & Search**: Encode chunks using embeddings and retrieve relevant contexts
- 🤖 **Answer Generation**: Generate answers from retrieved documents using LLMs

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/amnva/rag-docxplorer.git
cd rag-docxplorer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the RAG pipeline:
```bash
python main.py
```

## 📂 Data

Place your PDFs in a `data/` folder.
