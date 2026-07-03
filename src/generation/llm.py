# src/generation/llm.py
import re
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from src.config import Config
from langfuse import observe
from src.utils.telemetry import langfuse

class GenerationEngine:
    def __init__(self, confidence_threshold: float = 0.05):
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        self.embed_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        # Threshold to short-circuit if retrieval quality is too poor
        self.confidence_threshold = confidence_threshold

    def _build_context_block(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a strict, identifiable structure for the LLM."""
        context_str = ""
        for idx, chunk in enumerate(chunks):
            doc_name = chunk.get('document_name', f'Doc_{idx}')
            # We explicitly label the source to force accurate citation tagging
            context_str += f"--- SOURCE: [{doc_name}] ---\n{chunk['text']}\n\n"
        return context_str

    def _validate_citations(self, response_text: str, source_chunks: List[Dict[str, Any]]) -> Tuple[str, bool]:
        """
        Mathematically verifies that generated sentences match the cited source.
        Splits by sentence, finds citations, and runs a Cosine Similarity check.
        """
        # Split output into sentences (simplistic regex for demonstration)
        sentences = re.split(r'(?<=[.!?]) +', response_text)
        validated_response = []
        is_fully_confident = True

        for sentence in sentences:
            # Look for citations matching the pattern: [Document Name]
            citations = re.findall(r'\[([^\]]+)\]', sentence)
            
            if not citations:
                validated_response.append(sentence)
                continue

            sentence_is_valid = False
            for cited_doc in citations:
                # Find the actual chunk text that matches this citation
                matching_chunk = next((c['text'] for c in source_chunks if c['document_name'] == cited_doc), None)
                
                if matching_chunk:
                    # Mathematically verify the claim using our local embedding model
                    sent_vec = self.embed_model.encode([sentence])[0]
                    chunk_vec = self.embed_model.encode([matching_chunk])[0]
                    
                    similarity = np.dot(sent_vec, chunk_vec) / (np.linalg.norm(sent_vec) * np.linalg.norm(chunk_vec))
                    
                    # If semantic similarity is > 0.45, we consider the assertion mathematically grounded
                    if similarity > 0.45:
                        sentence_is_valid = True
                        break
            
            if sentence_is_valid:
                validated_response.append(sentence)
            else:
                # The LLM hallucinated a citation for a fact not in the source text
                is_fully_confident = False
                # Strip the fake citation and flag it inline
                cleaned_sentence = re.sub(r'\[([^\]]+)\]', '', sentence)
                validated_response.append(cleaned_sentence.strip() + " ⚠️ [UNVERIFIED ASSERTION]")

        final_text = " ".join(validated_response)
        if not is_fully_confident:
            final_text = "> **LOW CONFIDENCE WARNING:** Some statements could not be mathematically traced back to the source documents.\n\n" + final_text

        return final_text, is_fully_confident

    @observe(as_type="generation", name="Groq LLM Synthesis")
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Orchestrates degradation checks, prompt injection, and generation."""
        
        # 1. Graceful Degradation Check
        if not retrieved_chunks:
            return {"response": "I cannot locate verifiable data in the internal corpus to answer this question.", "confident": False}

        best_score = retrieved_chunks[0].get("rerank_score", 0)
        if best_score < self.confidence_threshold:
            print(f"📉 Highest retrieval score ({best_score:.4f}) fell below threshold. Triggering graceful degradation.")
            return {"response": "I cannot locate verifiable data in the internal corpus to answer this question.", "confident": False}

        # 2. Context Injection
        context_block = self._build_context_block(retrieved_chunks)
        
        system_prompt = (
            "You are a strict, enterprise data assistant. "
            "You must answer the user's query ONLY using the provided SOURCE texts below. "
            "If the answer is not contained in the sources, you must say 'I do not know.' "
            "For every fact or claim you output, you MUST append an inline citation matching the exact SOURCE name in brackets. "
            "Example: The server requires TLSv1.3 [sys_specs_2026.pdf].\n\n"
            f"AVAILABLE CONTEXT:\n{context_block}"
        )

        try:
            # 3. LLM Generation via Groq (Free & Lightning Fast)
            print("🧠 Generating answer via Groq LLM...")
            completion = self.groq_client.chat.completions.create(
                model=Config.TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1, # Extremely low temperature for deterministic facts
                max_tokens=500
            )
            raw_response = completion.choices[0].message.content

            # 4. Post-Generation Citation Validation
            print("🔬 Running mathematical citation verification...")
            validated_text, is_confident = self._validate_citations(raw_response, retrieved_chunks)

            
            validated_text, is_confident = self._validate_citations(raw_response, retrieved_chunks)
            
            if not is_confident:
                langfuse.update_current_trace(
                    tags=["hallucination_flag", "low_confidence"]
                )
            
        
            return {
                "response": validated_text,
                "confident": is_confident,
                "raw_context_used": retrieved_chunks
            }

        except Exception as e:
            print(f"⚠️ Generation failure: {str(e)}")
            return {"response": "A system error occurred during answer generation.", "confident": False}