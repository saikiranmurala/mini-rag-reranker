import os, json, sqlite3
from PyPDF2 import PdfReader
from tqdm import tqdm

def get_paragraphs_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    paragraphs = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraphs.extend(parts)
    return paragraphs

def main():
    PDF_DIR = 'data/public_pdfs'
    DB = 'chunks.sqlite'
    SOURCES_PATH = 'data/sources.json'

    if os.path.exists(DB):
        os.remove(DB)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE chunks (
        id INTEGER PRIMARY KEY, doc_id TEXT, chunk_id INTEGER,
        text TEXT, title TEXT, url TEXT
    );''')

    with open(SOURCES_PATH) as f:
        sources = json.load(f)

    for doc in tqdm(sources):
        doc_path = os.path.join(PDF_DIR, doc["pdf_filename"])
        paras = get_paragraphs_from_pdf(doc_path)
        chunk_id = 0
        for para in paras:
            words = para.split()
            if not (80 < len(words) < 500):
                continue
            c.execute("INSERT INTO chunks (doc_id, chunk_id, text, title, url) VALUES (?, ?, ?, ?, ?)",
                (doc["id"], chunk_id, para, doc["title"], doc["url"]))
            chunk_id += 1
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
