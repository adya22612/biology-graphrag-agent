import chromadb

def check_database():
    print("🔍 Checking local ChromaDB storage...\n")
    
    # 1. Connect directly to your local folder
    client = chromadb.PersistentClient(path="./chroma_db_storage")
    
    # 2. List all collections found in that folder
    collections = client.list_collections()
    
    if not collections:
        print("❌ No collections found. The database is empty.")
        return
        
    print(f"✅ Found {len(collections)} collection(s).")
    
    # Print the name of each collection
    for col in collections:
        # Handle different ChromaDB version return types
        col_name = getattr(col, 'name', col)
        print(f"   - {col_name}")
        
    # 3. Check the exact record count for your biology collection
    try:
        collection = client.get_collection(name="biology_textbook_vectors")
        count = collection.count()
        print(f"\n📈 Success! 'biology_textbook_vectors' contains {count} embeddings.")
    except Exception as e:
        print(f"\n⚠️ Could not load the specific biology collection: {e}")

if __name__ == "__main__":
    check_database()