# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    
    # Free Production Tier Models (2026 Standouts)
    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" 
    TEXT_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # Top open-source local embedding model
    
    # Storage Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGE_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "extracted_images")

    # Langfuse Keys
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")

    @classmethod
    def validate(cls):
        """Fails fast at runtime if environment variables are misconfigured."""
        missing = []
        if not cls.GROQ_API_KEY: missing.append("GROQ_API_KEY")
        if not cls.LLAMA_CLOUD_API_KEY: missing.append("LLAMA_CLOUD_API_KEY")
        if not cls.LANGFUSE_PUBLIC_KEY: missing.append("LANGFUSE_PUBLIC_KEY")
        if not cls.LANGFUSE_SECRET_KEY: missing.append("LANGFUSE_SECRET_KEY")
        if missing:
            raise ValueError(f"❌ Missing critical environment variables: {', '.join(missing)}")
        
        # Ensure directories exist
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.IMAGE_OUTPUT_DIR, exist_ok=True)

Config.validate()