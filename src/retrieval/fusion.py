# src/retrieval/fusion.py
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from langfuse import observe

class HybridFusionEngine:
    def __init__(self, k: int = 60):
        self.k = k
        # Completely free local cross-encoder model to act as our high-precision reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def reciprocal_rank_fusion(
        self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Blends rank variants across distinct engine passes using stabilizing constants."""
        rrf_scores = {}
        text_to_doc_map = {}

        # Process Dense Track ranks
        for rank, item in enumerate(dense_results):
            text = item["text"]
            text_to_doc_map[text] = item["document_name"]
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
            rrf_scores[text] += 1.0 / (self.k + (rank + 1))

        # Process Sparse Track ranks
        for rank, item in enumerate(sparse_results):
            text = item["text"]
            text_to_doc_map[text] = item["document_name"]
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
            rrf_scores[text] += 1.0 / (self.k + (rank + 1))

        # Reconstruct structural items list sorted by highest merged rank
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for text, score in sorted_docs:
            fused_results.append({
                "text": text,
                "document_name": text_to_doc_map[text],
                "rrf_score": score
            })
        return fused_results

    @observe(as_type="span", name="Cross-Encoder Reranking")
    def cross_encoder_rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Performs deep sentence pair sequence matching to compress context windows down."""
        if not candidate_chunks:
            return []

        # Prepare sentence pairs for deep cross-encoder evaluation
        pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
        scores = self.reranker.predict(pairs)

        # Inject re-calculated scores back into payload
        for idx, score in enumerate(scores):
            candidate_chunks[idx]["rerank_score"] = float(score)

        # Sort based strictly on high contextual validation match profiles
        reranked = sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]