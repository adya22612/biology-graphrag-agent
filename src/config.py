from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    # Database Configs
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str
    CHROMA_PATH: str = "./chroma_db_storage"
    
    # Postgres Checkpointer Config (Matches docker-compose)
    POSTGRES_URI: str 
    
    # LLM Config
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # LangSmith Tracing
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_PROJECT: str = "Biology_GraphRAG_V1"
    LANGSMITH_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Initialize it once to be imported anywhere in the app
settings = Settings()

os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY