import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from src.config import settings
from src.graph.state import AgentState
from src.graph.nodes import (
    retrieve_vector_context, evaluate_sufficiency,
    retrieve_graph_context, synthesize_answer, apply_guardrail
)

logger = logging.getLogger(__name__)

def route_after_evaluation(state: AgentState) -> str:
    return "synthesize_answer" if state.get("is_sufficient") == "yes" else "retrieve_graph_context"

def route_after_guardrail(state: AgentState) -> str:
    return END if state.get("is_sufficient") == "passed" else "synthesize_answer"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve_vector_context", retrieve_vector_context)
workflow.add_node("evaluate_sufficiency", evaluate_sufficiency)
workflow.add_node("retrieve_graph_context", retrieve_graph_context)
workflow.add_node("synthesize_answer", synthesize_answer)
workflow.add_node("apply_guardrail", apply_guardrail)

workflow.add_edge(START, "retrieve_vector_context")
workflow.add_edge("retrieve_vector_context", "evaluate_sufficiency")
workflow.add_conditional_edges("evaluate_sufficiency", route_after_evaluation, {"synthesize_answer": "synthesize_answer", "retrieve_graph_context": "retrieve_graph_context"})
workflow.add_edge("retrieve_graph_context", "synthesize_answer")
workflow.add_edge("synthesize_answer", "apply_guardrail")
workflow.add_conditional_edges("apply_guardrail", route_after_guardrail, {END: END, "synthesize_answer": "synthesize_answer"})

# We do not compile the agent here globally anymore. 
# In an async web server, we compile it during the FastAPI startup event.