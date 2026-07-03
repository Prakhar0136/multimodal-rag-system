# test_retrieval.py
from src.retrieval.dense import DenseRetrievalEngine
from src.retrieval.sparse import SparseRetrievalEngine
from src.retrieval.fusion import HybridFusionEngine

# Mock structured corpus parsed out of a system deployment schematic sheet
mock_chunks = [
    "System Deployment Config: Production Database cluster runs on node aws-db-cluster-prod-01. Host configuration requires active TLSv1.3 authorization.",
    "Financial metrics overview: Q3 gross margin expanded to 64.2% across global manufacturing verticals, driving standard net incomes higher.",
    "Error lookup table references: Internal error code ERR_404_AUTH_FAIL indicates missing cryptographic authorization handshake tokens.",
    "Architecture overview: System utilizes asynchronous microservices routed through high-throughput edge proxies processing standard API JSON footprints."
]

doc_name = "sys_specs_2026.pdf"

# Initialize Engines
dense_engine = DenseRetrievalEngine()
sparse_engine = SparseRetrievalEngine()
fusion_engine = HybridFusionEngine()

# Index the mock chunks into both search engines
dense_engine.index_chunks(mock_chunks, doc_name)
sparse_engine.fit(mock_chunks, doc_name)

# 🧪 Run a targeted query looking for an exact error string token
query = "What does the log error string ERR_404_AUTH_FAIL mean?"

print(f"\n🔍 Executing Search for query: '{query}'")
dense_res = dense_engine.search(query, limit=3)
sparse_res = sparse_engine.search(query, limit=3)

# Pass hits through the Fusion mechanism
fused_res = fusion_engine.reciprocal_rank_fusion(dense_res, sparse_res)
final_hits = fusion_engine.cross_encoder_rerank(query, fused_res, top_n=2)

print(f"\n🏆 Final Selected Retrieval Chunks after RRF & Cross-Encoder Compression:")
for idx, hit in enumerate(final_hits):
    print(f"[{idx+1}] Doc: {hit['document_name']} | Rerank Score: {hit['rerank_score']:.4f}")
    print(f"    Content: {hit['text']}\n")