# src/retrieval/dense.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from src.config import Config
from typing import List, Dict, Any

class DenseRetrievalEngine:
# Update the __init__ method in src/retrieval/dense.py
    def __init__(self, collection_name: str = "multimodal_chunks"):
        self.collection_name = collection_name
        
        # Read environment variables for cloud routing
        qdrant_host = os.getenv("QDRANT_HOST")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_host:
            # Production Cloud Connection
            self.client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
        else:
            # Fallback to local Docker
            self.client = QdrantClient(host="localhost", port=6333)
            
        self.embed_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        self.vector_size = 384 
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the collection in Qdrant if it hasn't been initialized yet."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            print(f"📦 Initializing Qdrant collection: '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, 
                    distance=Distance.COSINE
                )
            )

    def index_chunks(self, chunks: List[str], document_name: str):
        """Generates embeddings locally and writes payloads to Qdrant."""
        if not chunks:
            return
            
        print(f"🧬 Generating local dense embeddings for {len(chunks)} nodes...")
        embeddings = self.embed_model.encode(chunks)
        
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={
                        "text": chunk,
                        "document_name": document_name,
                        "chunk_index": idx
                    }
                )
            )
            
        # Upsert payload packets to local container storage
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ Successfully indexed dense vectors into Qdrant.")

    
    def search(self, query: str, limit: int = 10):
        """Queries Qdrant for the top N structurally closest chunks via cosine similarity."""
        query_vector = self.embed_model.encode(query).tolist()

        response = self.client.query_points(
        collection_name=self.collection_name,
        query=query_vector,
        limit=limit,
        )

        search_results = []
        for point in response.points:
            search_results.append({
                "text": point.payload["text"],
                "document_name": point.payload["document_name"],
                "score": point.score,
            })

        return search_results