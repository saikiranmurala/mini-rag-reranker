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

  
