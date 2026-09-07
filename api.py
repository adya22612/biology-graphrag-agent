import contextlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import settings
from src.graph.builder import workflow



# Define request schema
class ChatRequest(BaseModel):
    thread_id: str
    query: str

# Global variables for async resources
connection_pool = None
rag_agent = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles async setup and teardown for the API."""
    global connection_pool, rag_agent
    
    # 1. Startup: Connect to Postgres and Compile Agent
    connection_pool = AsyncConnectionPool(conninfo=settings.POSTGRES_URI,kwargs={"autocommit": True},open=False)
    await connection_pool.open()
    
    checkpointer = AsyncPostgresSaver(connection_pool)
    await checkpointer.setup()
    
    rag_agent = workflow.compile(checkpointer=checkpointer)
    print("🚀 FastAPI Server and LangGraph Agent Started.")
    
    yield
    
    # 2. Shutdown: Close connections cleanly
    await connection_pool.close()
    print("🛑 FastAPI Server Shut Down.")

app = FastAPI(title="Biology GraphRAG API", lifespan=lifespan)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """The REST endpoint the frontend talks to."""
    if not rag_agent:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    config = {"configurable": {"thread_id": request.thread_id}}
    initial_state = {
        "query": request.query,
        "generation_attempts": 0,
        "vector_context": "", "graph_context": "", "guardrail_feedback": "", "final_answer": ""
    }

    # Execute the graph asynchronously
    final_state = await rag_agent.ainvoke(initial_state, config=config)
    
    return {
        "answer": final_state.get("final_answer"),
        "metrics": {
            "attempts": final_state.get("generation_attempts"),
            "vector_sufficient": final_state.get("is_sufficient")
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)