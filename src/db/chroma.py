import logging
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.config import settings

logger = logging.getLogger(__name__)

chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    vector_collection = chroma_client.get_collection(
        name="biology_textbook_vectors",
        embedding_function=embedding_function
    )
except Exception as e:
    logger.error(f"Failed to load ChromaDB collection: {e}")
    raise