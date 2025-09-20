from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union
import numpy as np
from qa.baseline_search import search
from qa.rerank import hybrid_rerank

app = FastAPI()

# Request schema
class QARequest(BaseModel):
    q: str
    k: int = 3
    mode: str = "hybrid"  # "baseline" or "hybrid"

# Context chunk metadata returned
class ContextChunk(BaseModel):
    chunk_id: int
    text: str
    title: str
    url: str
    score: float
    final_score: Optional[float] = None
    vector_score: Optional[float] = None
    kw_score: Optional[float] = None

# Response schema
class QAResponse(BaseModel):
    answer: Optional[str]
    contexts: List[ContextChunk]
    reranker_used: bool

def answer_from_chunks(q, candidates, min_final_score=0.4) -> Optional[str]:
    if not candidates:
        return None
    top = candidates[0]
    if top.get("final_score", 0) < min_final_score:
        return None
    snippet = " ".join(top["text"].split()[:40]) + "..."
    cite = f"[{top['title']}]({top['url']})"
    return f"{snippet} {cite}"

@app.post("/ask", response_model=QAResponse)
def ask(req: QARequest):
    candidates = search(req.q, k=15)
    if req.mode == "hybrid":
        candidates = hybrid_rerank(req.q, candidates)
        rerank = True
    else:
        rerank = False
        for c in candidates:
            for score_field in ["score", "final_score", "vector_score", "kw_score"]:
                if score_field in c and isinstance(c[score_field], (np.float32, np.float64)):
                    c[score_field] = float(c[score_field])
    # Add display string
            c["final_score_display"] = f"The final score is {c.get('final_score', 0):.4f}"

    candidates = sorted(candidates, key=lambda x: -x["final_score"])[:req.k]

    # Convert numpy.float to float for serialization
    for c in candidates:
        for score_field in ["score", "final_score", "vector_score", "kw_score"]:
            if score_field in c and isinstance(c[score_field], (np.float32, np.float64)):
                c[score_field] = float(c[score_field])

    answer = answer_from_chunks(req.q, candidates)

    return {"answer": answer, "contexts": candidates, "reranker_used": rerank}
