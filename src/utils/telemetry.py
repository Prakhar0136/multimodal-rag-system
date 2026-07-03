import os
from langfuse import get_client
from src.config import Config

# Environment variables are used by the Langfuse client
os.environ["LANGFUSE_PUBLIC_KEY"] = Config.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = Config.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = os.getenv(
    "LANGFUSE_HOST",
    "https://cloud.langfuse.com",
)

langfuse = get_client()


def get_langfuse_client():
    return langfuse