# app.py
import streamlit as st
import requests

# UI Configuration
st.set_page_config(page_title="Enterprise Multimodal RAG", page_icon="🚀", layout="wide")
st.title("🚀 Enterprise Multimodal RAG Dashboard")
st.markdown("### Powered by Hybrid Retrieval, Cross-Encoder Reranking, and Llama 3")

# Sidebar for Observability & Settings
with st.sidebar:
    st.header("⚙️ Telemetry & Ops")
    st.markdown("[View Live Traces in Langfuse](https://cloud.langfuse.com)")
    st.info("The system automatically downgrades to 'Low Confidence' if retrieved data lacks mathematically verified assertions.")

st.divider()

# Main Chat Interface
query = st.text_input("Ask a question based on the indexed system documents:", placeholder="e.g., What does the ERR_404_AUTH_FAIL code mean?")

if st.button("Search & Synthesize"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Executing Hybrid Search & Verification Pipeline..."):
            try:
                # Call our FastAPI Gateway
                response = requests.post(
                    "http://localhost:8000/v1/ask", 
                    json={"query": query, "user_id": "streamlit_user_01"}
                )
                response.raise_for_status()
                data = response.json()
                
                # Display Answer
                if not data.get("confident"):
                    st.error("⚠️ Graceful Degradation Triggered / Low Confidence Flag")
                else:
                    st.success("✅ Mathematically Verified Assertions")
                    
                st.markdown(f"### Answer\n{data['answer']}")
                
                # Display Citations & Context
                st.divider()
                st.markdown("### 📚 Grounded Context & Reranker Confidence")
                
                if data.get("context"):
                    for idx, ctx in enumerate(data["context"]):
                        with st.expander(f"Reference [{ctx['document_name']}] - RRF & Cross-Encoder Score: {ctx.get('rerank_score', 0):.4f}"):
                            st.markdown(f"**Raw Chunk Text:**\n\n```text\n{ctx['text']}\n```")
                else:
                    st.info("No context was retrieved. System degraded gracefully.")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Failed to connect to API. Is FastAPI running on port 8000?")