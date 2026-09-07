import json
import logging
import asyncio
from ollama import AsyncClient
from src.config import settings
from src.graph.state import AgentState
from src.db.chroma import vector_collection
from src.db.neo4j import async_graph_driver

logger = logging.getLogger(__name__)

MAX_GENERATION_ATTEMPTS = 3
ollama_async = AsyncClient(host=settings.OLLAMA_BASE_URL)

async def retrieve_vector_context(state: AgentState) -> dict:
    query = state["query"]
    logger.info(f"Async Vector Retrieval for: '{query}'")

    try:
        # Local ChromaDB is synchronous, so we offload it to a background thread
        results = await asyncio.to_thread(
            vector_collection.query,
            query_texts=[query],
            n_results=3
        )
        documents = results.get("documents", [[]])[0]
        context = "\n---\n".join(documents) if documents else "No relevant vector context found."
    except Exception as e:
        logger.error(f"Vector retrieval failed: {e}")
        context = "Vector retrieval failed."

    return {"vector_context": context}

async def evaluate_sufficiency(state: AgentState) -> dict:
    query = state["query"]
    vector_context = state.get("vector_context", "")

    prompt = f"""You are an expert biology evaluator. Is this context sufficient to answer the question?
Question: {query}
Context: {vector_context}
Respond ONLY with JSON: {{"sufficient": "yes"}} or {{"sufficient": "no"}}"""

    try:
        response = await ollama_async.chat(
            model=settings.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"].strip()
        is_sufficient = "yes" if "yes" in content.lower() else "no"
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        is_sufficient = "no"

    return {"is_sufficient": is_sufficient}

async def retrieve_graph_context(state: AgentState) -> dict:
    query = state["query"]
    query_terms = [word.strip(",.?!") for word in query.split() if len(word) > 3]

    cypher_query = """
    MATCH (c:TextChunk)
    WHERE ANY(term IN $terms WHERE toLower(c.text) CONTAINS toLower(term))
    RETURN c.text AS chunk_text
    LIMIT 3
    """
    graph_chunks = []
    try:
        # Using asynchronous context managers for Neo4j
        async with async_graph_driver.session() as session:
            result = await session.run(cypher_query, terms=query_terms)
            async for record in result:
                graph_chunks.append(record["chunk_text"])
        
        graph_context = "\n---\n".join(graph_chunks) if graph_chunks else "No relevant graph context found."
    except Exception as e:
        logger.error(f"Graph retrieval failed: {e}")
        graph_context = "Graph context retrieval failed."

    return {"graph_context": graph_context}

async def synthesize_answer(state: AgentState) -> dict:
    query = state["query"]
    combined_context = f"Vector Context:\n{state.get('vector_context')}\n\nGraph Context:\n{state.get('graph_context')}"
    attempts = state.get("generation_attempts", 0) + 1

    # The upgraded, highly-opinionated prompt
    prompt = f"""You are an expert, conversational AI biology assistant. 
Your goal is to answer the user's question accurately using ONLY the provided context.

CRITICAL FORMATTING RULES:
1. Speak directly to the user. 
2. NEVER use phrases like "The answer to the user's question is...", "Based on the context...", or "According to the provided text...". Just state the facts directly.
3. Use a helpful, professional, and engaging tone.
4. Structure your answer for readability. Use bolding for key terms, and bullet points if you are listing more than two items.
5. If the context does not contain the answer, politely state that the information isn't available in your current knowledge base.

Context:
{combined_context}

Question: {query}"""

    # If the guardrail kicked it back, append the correction instructions
    if state.get("guardrail_feedback"):
        prompt += f"\n\nCRITICAL FIX REQUIRED: The previous answer failed validation. Address this feedback immediately: {state.get('guardrail_feedback')}"

    try:
        response = await ollama_async.chat(
            model=settings.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response["message"]["content"]
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        answer = "I apologize, an error occurred while generating the final answer."

    return {"final_answer": answer, "generation_attempts": attempts}

async def apply_guardrail(state: AgentState) -> dict:
    query = state["query"]
    final_answer = state.get("final_answer", "")
    attempts = state.get("generation_attempts", 1)

    prompt = f"""Evaluate for hallucinations and accuracy.
Query: {query}
Answer: {final_answer}
Respond ONLY in JSON format: {{"passed": true, "feedback": ""}} or {{"passed": false, "feedback": "Fix details"}}"""

    try:
        response = await ollama_async.chat(
            model=settings.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"].strip()
        passed = True if "true" in content.lower() else False
        feedback = "Correction needed based on guardrail." if not passed else ""
    except Exception:
        passed, feedback = True, ""

    if not passed and attempts >= MAX_GENERATION_ATTEMPTS:
        passed, feedback = True, "Max retries reached."

    return {
        "guardrail_feedback": "" if passed else feedback,
        "is_sufficient": "passed" if passed else "retry" # Re-using field for routing
    }