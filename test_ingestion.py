# test_ingestion.py
from src.ingestion.parser import MultimodalParser
from src.ingestion.chunker import ProductionChunker

# 1. Initialize our structures
parser = MultimodalParser()
chunker = ProductionChunker()

# Place a messy evaluation test PDF inside your data/ folder before executing
try:
    # 2. Extract layout content and analyze artifacts
    raw_content = parser.process_document("sample.pdf")
    
    # 3. Process semantic chunk variances
    semantic_nodes = chunker.semantic_splitting(raw_content)
    final_nodes = chunker.deduplicate_chunks(semantic_nodes)
    
    print(f"\n Verification Success!")
    print(f"Total Enriched Semantics Created: {len(final_nodes)}")
    if final_nodes:
        print(f"Sample Node Entry Look:\n {final_nodes[0][:300]}...")

except Exception as e:
    print(f"❌ Verification Pipeline Halted: {str(e)}")