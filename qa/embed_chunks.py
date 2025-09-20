import sqlite3, numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DB = 'chunks.sqlite'
INDEX_FILE = 'faiss.index'
MODEL_NAME = 'all-MiniLM-L6-v2'

def main():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, text FROM chunks")
    rows = c.fetchall()

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([row[1] for row in rows], show_progress_bar=True)
    ids = [row[0] for row in rows]

    np.savez('chunk_ids.npz', ids=ids)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))
    faiss.write_index(index, INDEX_FILE)
    np.save('embeddings.npy', embeddings)
    print(f"Stored {len(rows)} chunk embeddings.")

if __name__ == "__main__":
    main()
