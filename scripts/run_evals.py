import os
import sys
import json
import uuid
import httpx
import asyncio
from dotenv import load_dotenv
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")
from langsmith import Client, aevaluate
from langsmith.evaluation import EvaluationResult
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.config import settings

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "http://localhost:8000/chat"
DATASET_NAME = "Biology_GraphRAG_Benchmark_v1"
DATASET_FILE = Path(__file__).parent.parent / "tests" / "eval_dataset.json"

client = Client()

# ==========================================
# 1. SETUP LANGSMITH DATASET
# ==========================================
def setup_dataset():
    """Uploads the local JSON dataset to LangSmith if it doesn't exist."""
    if not DATASET_FILE.exists():
        print(f"Dataset file not found at {DATASET_FILE}")
        sys.exit(1)

    with open(DATASET_FILE, "r") as f:
        data = json.load(f)

    # Check if dataset already exists to prevent duplicates
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists in LangSmith. Reusing it.")
        return client.read_dataset(dataset_name=DATASET_NAME)

    print(f"Creating new dataset '{DATASET_NAME}' in LangSmith...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Ground truth questions and answers for Biology GraphRAG evaluation."
    )

    for item in data:
        client.create_example(
            inputs={"question": item["question"]},
            outputs={"expected_answer": item["expected_answer"]},
            dataset_id=dataset.id
        )
    return dataset

# ==========================================
# 2. DEFINE THE TARGET AGENT
# ==========================================
async def target_agent(inputs: dict) -> dict:
    """Sends the question to the live FastAPI server and returns the answer."""
    question = inputs["question"]
    thread_id = str(uuid.uuid4()) # Unique thread for each eval question
    
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        try:
            response = await http_client.post(
                API_URL, 
                json={"thread_id": thread_id, "query": question}
            )
            response.raise_for_status()
            data = response.json()
            return {"answer": data.get("answer", "No answer provided.")}
        except Exception as e:
            return {"answer": f"API Error: {e}"}

# ==========================================
# 3. DEFINE THE LLM-AS-A-JUDGE EVALUATOR
# ==========================================
class GradeOutput(BaseModel):
    score: int = Field(description="Grade from 1 to 5")
    reasoning: str = Field(description="Why this score was given")

parser = PydanticOutputParser(pydantic_object=GradeOutput)
judge_llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0, format="json")

async def correctness_evaluator(run, example) -> EvaluationResult:
    """Evaluates factual accuracy against the ground truth."""
    agent_answer = run.outputs.get("answer", "")
    expected_answer = example.outputs.get("expected_answer", "")
    question = example.inputs.get("question", "")

    prompt = f"""You are a strict teacher grading a biology test.
Question: {question}
True Answer: {expected_answer}
Student Answer: {agent_answer}

Grade the Student Answer from 1 to 5 based on accuracy compared to the True Answer.
1 = Completely wrong, 5 = Perfect.

You MUST output ONLY valid JSON in this exact format:
{{
    "score": 5,
    "reasoning": "short explanation"
}}
"""
    try:
        response = await judge_llm.ainvoke(prompt)
        data = json.loads(response.content)
        score = int(data.get("score", 1))
        normalized_score = (score - 1) / 4.0 
    except Exception as e:
        normalized_score = 0.0
        data = {"reasoning": f"Parse failed: {e}"}

    return EvaluationResult(key="correctness", score=normalized_score, comment=data.get("reasoning"))


async def relevance_evaluator(run, example) -> EvaluationResult:
    """Evaluates if the answer directly addresses the user's prompt without rambling."""
    agent_answer = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    prompt = f"""You are evaluating an AI assistant.
User Question: {question}
AI Answer: {agent_answer}

Grade the AI Answer from 1 to 5 based on how directly and concisely it answered the specific question. Do not grade factual accuracy, only grade relevance.
1 = Completely off-topic or rambling, 5 = Direct, concise, and highly relevant.

You MUST output ONLY valid JSON in this exact format:
{{
    "score": 5,
    "reasoning": "short explanation"
}}
"""
    try:
        response = await judge_llm.ainvoke(prompt)
        data = json.loads(response.content)
        score = int(data.get("score", 1))
        normalized_score = (score - 1) / 4.0 
    except Exception as e:
        normalized_score = 0.0
        data = {"reasoning": f"Parse failed: {e}"}

    return EvaluationResult(key="relevance", score=normalized_score, comment=data.get("reasoning"))

# ==========================================
# 4. RUN EVALUATION
# ==========================================
async def main():
    print("Step 1: Setting up dataset...")
    dataset = setup_dataset()

    print(f"Step 2: Starting evaluation against FastAPI backend at {API_URL}...")
    
    results = await aevaluate(
        target_agent,
        data=DATASET_NAME,
        # Notice we are now passing an array of multiple evaluators!
        evaluators=[correctness_evaluator, relevance_evaluator],
        experiment_prefix="GraphRAG_MultiMetric_v1",
        description="Testing correctness and answer relevance simultaneously.",
    )
    print("\n🎉 Evaluation complete! View the full report card in your LangSmith dashboard.")

if __name__ == "__main__":
    asyncio.run(main())