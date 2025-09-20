import numpy as np
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
DB = 'chunks.sqlite'
INDEX_FILE = 'faiss.index'

def search(question, k=6):
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([question])
    index = faiss.read_index(INDEX_FILE)
    D, I = index.search(np.array(q_emb, dtype='float32'), k*2)

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    chunk_ids = np.load('chunk_ids.npz', allow_pickle=True)['ids']
    results = []
    for score, idx in zip(D[0], I[0]):
        chunk_id = chunk_ids[idx]
        c.execute("SELECT text, title, url FROM chunks WHERE id=?", (int(chunk_id),))
        row = c.fetchone()
        if row:
            results.append({'chunk_id': int(chunk_id), 'score': float(-score), 'text': row[0], 'title': row[1], 'url': row[2]})
    return results

if __name__ == "__main__":
    res = search("What is machine safety?", k=5)
    for r in res:
        print(r['score'], r['title'], r['text'][:80])
