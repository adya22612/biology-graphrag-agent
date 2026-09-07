import os
import sys
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root directory is in sys.path when running as a standalone script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from neo4j import GraphDatabase

from src.config import settings

# =========================================
# LOGGING CONFIGURATION
# =========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("IngestionPipeline")


# =========================================
# HELPER FUNCTIONS
# =========================================
def load_and_split_pdf_documents(folder_path: str) -> List[Document]:
    """Scans the specified directory for PDF files and splits them into chunks."""
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        raise FileNotFoundError(
            f"Directory '{folder_path}' does not exist or contains no files."
        )

    logger.info(f"Scanning '{folder_path}' for textbook PDFs...")
    raw_pages: List[Document] = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            logger.info(f"Loading document: {file}")
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            raw_pages.extend(pages)

    logger.info(f"Loaded {len(raw_pages)} raw document pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_pages)
    logger.info(f"Divided document content into {len(chunks)} text chunks.")
    return chunks


def batch_ingest_to_neo4j(session, batch: List[Dict[str, Any]]) -> None:
    """Executes a batched Cypher UNWIND query to write chunks into Neo4j efficiently."""
    query = """
    UNWIND $batch AS row
    MERGE (src:SourceFile {path: row.source_file})
    MERGE (page:PageNumber {num: row.page_num})
    MERGE (src)-[:HAS_PAGE]->(page)
    
    MERGE (chunk:TextChunk {id: row.chunk_id})
    SET chunk.text = row.chunk_text
    MERGE (page)-[:CONTAINS_TEXT]->(chunk)
    """
    session.run(query, batch=batch)


# =========================================
# MASTER INGESTION PIPELINE
# =========================================
def run_master_ingestion(textbook_folder: str = "./biology_textbook", batch_size: int = 50) -> None:
    """Orchestrates multi-database ingestion into ChromaDB and Neo4j."""
    try:
        chunks = load_and_split_pdf_documents(textbook_folder)
    except Exception as e:
        logger.error(f"Failed during PDF loading/splitting phase: {e}")
        return

    # 1. Initialize ChromaDB
    logger.info(f"Initializing persistent ChromaDB client at path: {settings.CHROMA_PATH}")
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    chroma_collection = chroma_client.get_or_create_collection(
        name="biology_textbook_vectors",
        embedding_function=embedding_function
    )

    # 2. Initialize Neo4j Driver
    logger.info(f"Connecting to Neo4j instance at: {settings.NEO4J_URI}")
    try:
        neo4j_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        neo4j_driver.verify_connectivity()
    except Exception as e:
        logger.error(f"Neo4j connection failure. Verify database credentials and status. Error: {e}")
        return

    # 3. Process and Ingest Data in Batches
    logger.info("Commencing batched ingestion loop into ChromaDB and Neo4j...")
    
    chroma_docs, chroma_metadatas, chroma_ids = [], [], []
    neo4j_batch = []

    with neo4j_driver.session() as session:
        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            source_file = os.path.basename(chunk.metadata.get("source", "Unknown_File.pdf"))
            page_num = int(chunk.metadata.get("page", 0)) + 1

            # Prepare Chroma Payload
            chroma_docs.append(chunk.page_content)
            chroma_metadatas.append({"source": source_file, "page": page_num, "chunk_id": chunk_id})
            chroma_ids.append(chunk_id)

            # Prepare Neo4j Batch Record
            neo4j_batch.append({
                "source_file": source_file,
                "page_num": page_num,
                "chunk_id": chunk_id,
                "chunk_text": chunk.page_content
            })

            # Execute batch insert when reaching batch_size or end of dataset
            if len(neo4j_batch) >= batch_size or (idx + 1) == len(chunks):
                # Write batch to ChromaDB
                chroma_collection.add(
                    documents=chroma_docs,
                    metadatas=chroma_metadatas,
                    ids=chroma_ids
                )
                
                # Write batch to Neo4j
                batch_ingest_to_neo4j(session, neo4j_batch)

                logger.info(f"Synchronized [{idx + 1}/{len(chunks)}] chunks across databases...")
                
                # Clear buffers
                chroma_docs.clear()
                chroma_metadatas.clear()
                chroma_ids.clear()
                neo4j_batch.clear()

    neo4j_driver.close()
    logger.info("🎉 Ingestion Pipeline successfully completed and connections closed.")


if __name__ == "__main__":
    # Ensure settings loads correctly before executing
    TEXTBOOK_DIR = os.getenv("TEXTBOOK_FOLDER", "./biology_textbook")
    run_master_ingestion(textbook_folder=TEXTBOOK_DIR)