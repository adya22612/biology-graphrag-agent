from typing import TypedDict

class AgentState(TypedDict):
    query: str
    vector_context: str
    graph_context: str
    is_sufficient: str
    final_answer: str
    guardrail_feedback: str 
    generation_attempts: int