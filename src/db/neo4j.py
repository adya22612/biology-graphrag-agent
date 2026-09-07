import logging
from neo4j import AsyncGraphDatabase
from src.config import settings

logger = logging.getLogger(__name__)

# Initialize the ASYNC driver
try:
    async_graph_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    logger.info("Async Neo4j driver initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Async Neo4j driver: {e}")
    raise