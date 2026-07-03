# src/retrieval/sparse.py
import re
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class SparseRetrievalEngine:
    def __init__(self):
        self.bm25 = None
        self.corpus = []  # Stores raw text mapping chunks
        self.metadata = [] # Stores document metadata associations

    def _tokenize(self, text: str) -> List[str]:
        """Lowercases and extracts raw alphanumerical tokens from strings."""
        return re.findall(r'\w+', text.lower())

    def fit(self, chunks: List[str], document_name: str):
        """Indexes the text chunks into the lexical BM25 database structures."""
        self.corpus.extend(chunks)
        for idx, chunk in enumerate(chunks):
            self.metadata.append({
                "document_name": document_name,
                "chunk_index": idx
            })
            
        tokenized_corpus = [self._tokenize(chunk) for chunk in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"📖 Lexical BM25 index built with {len(self.corpus)} unique items.")

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Queries the lexical corpus for strict keyword alignments."""
        if not self.bm25 or not self.corpus:
            return []
            
        tokenized_query = self._tokenize(query)
        # Scores all indexed nodes based on term frequency-inverse document frequency concepts
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices based on highest lexical relevance scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        
        search_results = []
        for idx in top_indices:
            # Skip items that have absolutely zero matched overlap tokens
            if scores[idx] == 0:
                continue
            search_results.append({
                "text": self.corpus[idx],
                "document_name": self.metadata[idx]["document_name"],
                "score": float(scores[idx])
            })
        return search_results