import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Literal

# Ensure project root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import chromadb
import ollama
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from src.config import settings

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SemanticBuilder")

COLLECTION_NAME = "biology_textbook_vectors"

# ==========================================
# STRICT PYDANTIC SCHEMAS
# ==========================================
class Entity(BaseModel):
    name: str
    type: str

class Relationship(BaseModel):
    source: str
    relation: str
    target: str

class GraphExtraction(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


# ==========================================
# DIRECT OLLAMA EXTRACTION (No LangChain Overhead)
# ==========================================
def extract_graph(chunk_text: str) -> GraphExtraction:
    """Uses direct Ollama JSON chat with a concise prompt tailored for 3B models."""
    
    prompt = f"""Extract biological entities and relationships from the text.

Rules:
1. Lowercase all entity names (e.g. "mitochondria", "atp").
2. Only extract facts explicitly in the text.
3. If no clear biological concepts exist in the text, return empty lists.

Allowed Entity Types: Concept, Process, Organism, Plant, Animal, Organ, Tissue, Cell, CellStructure, Molecule, Protein, Hormone, Gene, Nutrient
Allowed Relationships: REQUIRES, PRODUCES, PART_OF, CONTAINS, OCCURS_IN, CAUSES, DEVELOPS_INTO, FORMS, TRANSPORTS, LOCATED_IN, COMPOSED_OF, ABSORBS, RELEASES, USES, CONVERTS_TO, RESPONSIBLE_FOR, IS_A

Return strictly JSON format:
{{
  "entities": [
    {{"name": "mitochondria", "type": "CellStructure"}}
  ],
  "relationships": [
    {{"source": "mitochondria", "relation": "PRODUCES", "target": "atp"}}
  ]
}}

Text:
{chunk_text}"""

    try:
        response = ollama.chat(
            model=settings.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0}
        )
        
        raw_content = response["message"]["content"]
        # Validate against Pydantic schema
        return GraphExtraction.model_validate_json(raw_content)
        
    except Exception as e:
        logger.warning(f"Extraction error: {e}")
        return GraphExtraction()


# ==========================================
# NEO4J DATABASE LOGIC
# ==========================================
def check_already_processed(session, chunk_id: str) -> bool:
    result = session.run(
        "MATCH (c:TextChunk {id:$chunk_id}) RETURN c.processed AS processed",
        chunk_id=chunk_id
    ).single()
    return result is not None and result["processed"] is True

def insert_graph(session, graph_data: GraphExtraction, metadata: dict) -> None:
    chunk_id = metadata["chunk_id"]
    
    # 1. Chunk Node
    session.run(
        """
        MERGE (c:TextChunk {id:$chunk_id})
        SET c.source = $source, c.page = $page
        """,
        chunk_id=chunk_id,
        source=metadata.get("source", "Unknown"),
        page=metadata.get("page", 0)
    )

    # 2. Entities
    if graph_data.entities:
        entity_dicts = [{"name": e.name.strip().lower(), "type": e.type} for e in graph_data.entities if e.name]
        
        if entity_dicts:
            session.run(
                """
                UNWIND $entities AS ent
                MERGE (e:Entity {name: ent.name})
                SET e.type = ent.type
                WITH e
                MATCH (c:TextChunk {id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                entities=entity_dicts,
                chunk_id=chunk_id
            )

    # 3. Relationships
    valid_relations = {
        "REQUIRES", "PRODUCES", "PART_OF", "CONTAINS", "OCCURS_IN", 
        "CAUSES", "DEVELOPS_INTO", "FORMS", "TRANSPORTS", "LOCATED_IN", 
        "COMPOSED_OF", "ABSORBS", "RELEASES", "USES", "CONVERTS_TO", 
        "RESPONSIBLE_FOR", "IS_A"
    }
    
    for rel in graph_data.relationships:
        rel_type = rel.relation.upper().strip()
        if rel_type not in valid_relations:
            continue
            
        source_name = rel.source.strip().lower()
        target_name = rel.target.strip().lower()
        if not source_name or not target_name:
            continue
        
        query = f"""
        MATCH (a:Entity {{name:$source}})
        MATCH (b:Entity {{name:$target}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        session.run(query, source=source_name, target=target_name)

    # 4. Mark processed
    session.run("MATCH (c:TextChunk {id:$chunk_id}) SET c.processed = true", chunk_id=chunk_id)


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def run_semantic_extraction():
    logger.info(f"Connecting to ChromaDB at {settings.CHROMA_PATH}...")
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    
    logger.info(f"Loaded {len(documents)} chunks from ChromaDB.")

    driver = GraphDatabase.driver(
        settings.NEO4J_URI, 
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        for idx, (chunk_text, metadata) in enumerate(zip(documents, metadatas)):
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                continue

            if check_already_processed(session, chunk_id):
                logger.info(f"[{idx+1}/{len(documents)}] ⏭️ Already processed.")
                continue

            logger.info(f"[{idx+1}/{len(documents)}] Processing chunk (ID: {chunk_id[:8]})...")
            
            # Print sample text for first few chunks to verify Chroma data
            if idx < 3:
                logger.info(f"  Sample Text: {chunk_text[:120]}...")

            graph_data = extract_graph(chunk_text)
            insert_graph(session, graph_data, metadata)
            
            logger.info(f"  ✅ Extracted {len(graph_data.entities)} entities & {len(graph_data.relationships)} relationships.")

    driver.close()
    logger.info("🎉 Semantic Graph Construction Complete!")

if __name__ == "__main__":
    run_semantic_extraction()