# src/main_pipeline.py
from langfuse import observe
from src.retrieval.dense import DenseRetrievalEngine
from src.retrieval.sparse import SparseRetrievalEngine
from src.retrieval.fusion import HybridFusionEngine
from src.generation.llm import GenerationEngine
from src.utils.telemetry import langfuse

class MultimodalRAGPipeline:
    def __init__(self):
        self.dense = DenseRetrievalEngine()
        self.sparse = SparseRetrievalEngine()
        self.fusion = HybridFusionEngine()
        self.llm = GenerationEngine()

    @observe(as_type="span", name="Hybrid Retrieval Phase")
    def retrieve(self, query: str):
        """Wraps the retrieval logic into a single observable span."""
        dense_res = self.dense.search(query, limit=5)
        sparse_res = self.sparse.search(query, limit=5)
        fused = self.fusion.reciprocal_rank_fusion(dense_res, sparse_res)
        return self.fusion.cross_encoder_rerank(query, fused, top_n=3)

    @observe(name="POST /v1/ask Root Trace")
    def execute_query(self, query: str, user_id: str = "anonymous"):
        """The root trace that tracks the entire lifecycle of a user request."""
        # 1. Attach user metadata to the trace
        langfuse.update_current_trace(
            user_id=user_id,
            session_id="session_001",
        )
        
        # 2. Execute Retrieval
        context = self.retrieve(query)
        
        # 3. Execute Generation
        result = self.llm.generate_answer(query, context)
        
        # Force flush telemetry data to Langfuse cloud before returning
        langfuse.flush()
        
        return result