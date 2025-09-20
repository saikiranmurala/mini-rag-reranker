from rank_bm25 import BM25Okapi
import numpy as np

def bm25_scores(query, candidate_chunks):
    tokenized = [c["text"].split() for c in candidate_chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    return scores

def hybrid_rerank(query, baseline_results, alpha=0.6):
    vector_scores = np.array([r['score'] for r in baseline_results])
    if np.ptp(vector_scores):
        vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min())
    else:
        vector_scores = np.ones_like(vector_scores)

    bm25 = bm25_scores(query, baseline_results)
    if np.ptp(bm25):
        bm25 = (bm25 - np.min(bm25)) / (np.max(bm25) - np.min(bm25))
    else:
        bm25 = np.ones_like(bm25) * 0.5

    reranked = []
    for i, r in enumerate(baseline_results):
        final = alpha * vector_scores[i] + (1-alpha) * bm25[i]
        reranked.append({**r, "final_score": float(final), "vector_score": float(vector_scores[i]), "kw_score": float(bm25[i])})
    sorted_results = sorted(reranked, key=lambda x: -x["final_score"])
    return sorted_results
