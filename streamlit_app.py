import streamlit as st
from qa.baseline_search import search
from qa.rerank import hybrid_rerank
import numpy as np

def answer_from_chunks(q, candidates, min_final_score=0.4):
    if not candidates:
        return None
    top = candidates[0]
    if top['final_score'] < min_final_score:
        return None
    snippet = " ".join(top['text'].split()[:40]) + "..."
    cite = f"[{top['title']}]({top['url']})"
    return f"{snippet} {cite}"

def main():
    st.title("Mini RAG Q&A System with Hybrid Reranker")
    query = st.text_area("Ask a question about industrial/machine safety:", height=100)
    k = st.slider("Number of answers to return (k):", 1, 10, 3)
    mode = st.radio("Select mode:", ["baseline", "hybrid"], index=1)

    if st.button("Get Answers"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        candidates = search(query, k=15)
        if mode == "hybrid":
            candidates = hybrid_rerank(query, candidates)
        else:
            for c in candidates:
                c["final_score"] = float(c["score"])
            candidates = sorted(candidates, key=lambda x: -x["final_score"])
        candidates = candidates[:k]

        # Ensure numeric fields are native python float
        for c in candidates:
            for f in ["score", "final_score", "vector_score", "kw_score"]:
                if f in c and isinstance(c[f], (np.float32, np.float64)):
                    c[f] = float(c[f])

        answer = answer_from_chunks(query, candidates)
        if answer:
            st.markdown(f"### Answer:\n{answer}")
        else:
            st.info("No confident answer found. Try rephrasing or broadening your question.")

        with st.expander("Context chunks and scores"):
            for i, c in enumerate(candidates):
                st.markdown(f"**Context {i+1}** (final_score={c.get('final_score', None):.3f}):")
                st.write(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
                st.markdown(f"[Source: {c['title']}]({c['url']})")
                st.markdown("---")

if __name__ == "__main__":
    main()
