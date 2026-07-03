# test_generation.py
from src.generation.llm import GenerationEngine

engine = GenerationEngine()

# Mocking the output we would have gotten from Phase 2 (RRF + Reranker)
mock_retrieved_context = [
    {
        "text": "The primary database cluster runs on AWS Node aws-db-cluster-prod-01 and requires active TLSv1.3 authorization.",
        "document_name": "sys_specs_2026.pdf",
        "rerank_score": 0.85 # High confidence
    }
]

print("--- Test 1: Valid Query ---")
query_1 = "What node does the database run on, and what auth is required?"
result_1 = engine.generate_answer(query_1, mock_retrieved_context)
print(f"Result:\n{result_1['response']}\n")

print("--- Test 2: Graceful Degradation (Irrelevant Query) ---")
# To simulate degradation, we pass an empty context or a context with a terrible rerank score
mock_bad_context = [
    {
        "text": "The company holiday party is scheduled for Friday.",
        "document_name": "hr_memo.pdf",
        "rerank_score": 0.01 # Below our 0.05 threshold
    }
]
query_2 = "What is the capital of France?"
result_2 = engine.generate_answer(query_2, mock_bad_context)
print(f"Result:\n{result_2['response']}\n")