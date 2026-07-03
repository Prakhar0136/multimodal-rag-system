# test_observability.py
from src.main_pipeline import MultimodalRAGPipeline
from langfuse.decorators import langfuse_context  # <-- Add this import

pipeline = MultimodalRAGPipeline()

print("🚀 Firing a request through the fully instrumented pipeline...")
result = pipeline.execute_query(
    query="What does the log error string ERR_404_AUTH_FAIL mean?",
    user_id="test_user_99"
)

print("\n✅ Execution Complete.")
print(f"Response: {result['response']}")

# <-- ADD THIS LINE: Force the background thread to upload data before Python exits
langfuse_context.flush() 

print("\n👉 Now, go log into your Langfuse dashboard and click on 'Traces'.")

