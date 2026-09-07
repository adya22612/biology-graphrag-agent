import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# PATH FIX: TELL PYTHON WHERE 'src' LIVES
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Load environment variables
load_dotenv(ROOT_DIR / ".env")

from langsmith import Client

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_NAME = os.environ.get("LANGCHAIN_PROJECT", "Biology_GraphRAG_Prod")
client = Client()

def main():
    print(f"📊 Fetching recent chat traces for project: {PROJECT_NAME}...\n")
    
    try:
        # Fetch the last 50 main chat interactions
        root_runs = list(client.list_runs(
            project_name=PROJECT_NAME,
            is_root=True,
            error=False,
            limit=50
        ))
    except Exception as e:
        print(f"❌ Failed to fetch runs: {e}")
        return

    if not root_runs:
        print("No runs found. Chat with your agent first!")
        return

    routing_data = []

    # Iterate through each chat to analyze the LangGraph routing
    for root in root_runs:
        # Fetch all the individual nodes (child runs) that executed during this chat
        trace_runs = list(client.list_runs(trace_id=root.id))
        
        # Extract the names of all nodes that were triggered
        executed_nodes = [r.name for r in trace_runs if r.name]
        
        # Analyze the specific path the agent took
        used_vector = "retrieve_vector_context" in executed_nodes
        used_graph = "retrieve_graph_context" in executed_nodes
        hit_guardrail = "apply_guardrail" in executed_nodes
        
        # Determine the primary route
        if used_graph:
            route = "Graph Database (Fallback)"
        elif used_vector:
            route = "Vector Database (Primary)"
        else:
            route = "Unknown/Error"

        routing_data.append({
            "trace_id": str(root.id),
            "route_taken": route,
            "hit_guardrail": hit_guardrail,
            "latency_seconds": (root.end_time - root.start_time).total_seconds() if root.end_time else 0
        })

    # Load into Pandas for aggregation and analysis
    df = pd.DataFrame(routing_data)

    # Calculate statistics
    total_chats = len(df)
    route_counts = df['route_taken'].value_counts()
    route_percentages = df['route_taken'].value_counts(normalize=True) * 100
    guardrail_triggers = df['hit_guardrail'].sum()

    # Print the Executive Summary
    print("==================================================")
    print("📈 LANGGRAPH ROUTING ANALYTICS REPORT")
    print("==================================================")
    print(f"Total Interactions Analyzed: {total_chats}\n")
    
    print("🚦 Routing Distribution:")
    for route, count in route_counts.items():
        percentage = route_percentages[route]
        print(f"  - {route}: {count} chats ({percentage:.1f}%)")
        
    print(f"\n🛡️ Guardrail Interventions: {guardrail_triggers} times ({(guardrail_triggers/total_chats)*100:.1f}%)")
    
    print(f"⏱️ Average Latency: {df['latency_seconds'].mean():.2f} seconds")
    print("==================================================")

if __name__ == "__main__":
    main()