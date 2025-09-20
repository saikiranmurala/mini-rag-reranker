# Industrial Safety Mini RAG Q&A System

## Overview

This project implements a small question-answering system over 20 industrial & machine safety documents.  
It builds a retriever augmented generator (RAG) style extractive QA system with:

- Chunking PDFs into paragraphs stored in SQLite  
- Embedding chunks using Sentence Transformers (all-MiniLM-L6-v2)  
- Vector similarity search with FAISS  
- Hybrid reranker blending vector score and BM25 keyword matches  
- FastAPI endpoint `/ask` for question answering with citations  
- Optional Streamlit frontend for interactive Q&A  

  

## Overview

This project implements a small, transparent Retrieval-Augmented Generation (RAG) system to answer questions over 20 industrial and machine safety documents. It ingests public PDFs, chunks them into paragraphs stored in SQLite, computes dense embeddings, and indexes them with FAISS. A baseline cosine similarity search is enhanced by a hybrid reranker combining semantic and keyword (BM25) matching. Answers are extractive with strong provenance, supporting trustworthy, evidence-based Q&A.

---

## Setup Instructions

1. **Clone this repository** and navigate into the project directory.

2. **Install required Python packages**:

   ```
   pip install -r requirements.txt
   pip install pycryptodome huggingface_hub==0.13.4
   ```

3. **Place the 20 PDF files** into `data/public_pdfs/`, and ensure `data/sources.json` correctly lists them with consistent filenames and metadata.

4. **Chunk the PDFs** into paragraphs and store them in SQLite:

   ```
   python qa/chunk_docs.py
   ```

5. **Generate embeddings and create the FAISS index**:

   ```
   python qa/embed_chunks.py
   ```

6. **Run the FastAPI server**:

   ```
   uvicorn qa.api:app --host 127.0.0.1 --port 8000
   ```

7. (Optional) **Run the Streamlit frontend for interactive Q&A**:

   ```
   streamlit run streamlit_app.py
   ```

---

## How to Use

Send POST requests with JSON body `{ "q": "question", "k": 3, "mode": "hybrid" }` to:

```
http://127.0.0.1:8000/ask
```

Example curl commands:

Easy question:
'''
 What is PLd?", "k":2, "mode":"hybrid"
'''

Tricky question:

```

     Who must document the PLd calculation for a new press line?", "k":3, "mode":"hybrid"
```

---



## Learnings

Building this mini RAG pipeline from scratch provided deep insight into the challenges of document chunking, vector embedding, indexing, and especially balancing semantic similarity with keyword precision in industrial safety texts. The hybrid reranker was essential to boost factual precision, a key for trustworthy industrial QA systems. The project also reinforced the importance of clear citations and abstaining when confidence is low to avoid misleading answers.

