# src/ingestion/chunker.py
import re
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.config import Config

class ProductionChunker:
    def __init__(self):
        # Local, free, lightning-fast embedding compute engine
        self.embed_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)

    def fixed_size_chunking(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Classic sliding window approach based on characters/words representation."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
        return chunks

    def structure_aware_chunking(self, text: str) -> List[str]:
        """Splits strictly on Markdown structural section lines (#, ##, ###)."""
        # Matches markdown headers while maintaining structural anchor points
        header_pattern = r'(?==+)?\s*\n(?:#+\s+.*)'
        chunks = re.split(header_pattern, text)
        return [c.strip() for c in chunks if c.strip()]

    def semantic_splitting(self, text: str, alpha: float = 1.2) -> List[str]:
        """Dynamically breaks text bounds when adjacent sentence embedding distances spike."""
        # Simple regex tokenizer splitting on sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            return sentences

        embeddings = self.embed_model.encode(sentences)
        
        # Calculate cosine distances between adjacent sentences
        distances = []
        for i in range(len(embeddings) - 1):
            vec1 = embeddings[i]
            vec2 = embeddings[i+1]
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            distances.append(1.0 - similarity)

        # Set threshold dynamically via statistical deviation
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + (alpha * std_dist)

        chunks = []
        current_chunk = sentences[0]

        for i, distance in enumerate(distances):
            if distance > threshold:
                chunks.append(current_chunk)
                current_chunk = sentences[i+1]
            else:
                current_chunk += " " + sentences[i+1]
        
        chunks.append(current_chunk)
        return chunks

    def deduplicate_chunks(self, chunks: List[str], similarity_threshold: float = 0.95) -> List[str]:
        """Matrix de-duplication pass. Prunes overlapping nodes to minimize LLM token bloat."""
        if not chunks:
            return []
            
        embeddings = self.embed_model.encode(chunks)
        unique_chunks = []
        kept_indices = []

        for i, vec_i in enumerate(embeddings):
            is_duplicate = False
            for j in kept_indices:
                vec_j = embeddings[j]
                similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_chunks.append(chunks[i])
                kept_indices.append(i)

        print(f"✂️ Pruned {len(chunks) - len(unique_chunks)} near-duplicate chunks out of the stream.")
        return unique_chunks