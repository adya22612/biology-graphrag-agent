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
OUTPUT_FILE = ROOT_DIR / "tests" / "curated_golden_dataset.csv"

client = Client()

def main():
    print(f"Fetching runs for project: {PROJECT_NAME}...")
    
    # 1. Fetch Runs from LangSmith
    # We filter to only get root runs (the main chat interaction) and ignore errors
    try:
        runs = client.list_runs(
            project_name=PROJECT_NAME,
            is_root=True,
            error=False 
        )
    except Exception as e:
        print(f"❌ Failed to fetch runs: {e}")
        return
    
    # 2. Extract Data 
    data = []
    for run in runs:
        if not run.inputs or not run.outputs:
            continue
            
        # LangGraph inputs/outputs are dictionaries
        question = run.inputs.get("query", "")
        # Adjust 'final_answer' if your state dictionary uses a different key
        answer = run.outputs.get("final_answer", "") 
        
        if question and answer:
            data.append({
                "question": question,
                "expected_answer": answer,
                "trace_id": str(run.id),
                "timestamp": run.start_time
            })
            
    if not data:
        print("No valid runs found. Have you chatted with the agent via Streamlit yet?")
        return

    # 3. Clean and Transform with Pandas
    print(f"Extracted {len(data)} raw chat interactions. Loading into Pandas...")
    df = pd.DataFrame(data)
    
    # Data Cleaning: Drop exact duplicate questions from users asking the same thing twice
    df = df.drop_duplicates(subset=["question"])
    
    # Data Cleaning: Strip accidental whitespace
    df["question"] = df["question"].str.strip()
    df["expected_answer"] = df["expected_answer"].str.strip()
    
    # 4. Save the Curated Dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 Saved curated dataset to {OUTPUT_FILE}")
    print(f"Total unique, clean examples: {len(df)}")
    
    # Note: As this dataset scales to hundreds of rows, you could integrate 
    # Scikit-learn (e.g., using TF-IDF and KMeans) to automatically cluster 
    # these questions and identify missing topic coverage in your textbook!

if __name__ == "__main__":
    main()