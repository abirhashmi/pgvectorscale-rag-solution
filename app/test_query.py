
from database.vector_store import VectorStore # Initialize vector store
vec = VectorStore() # User-defined question
query = "What are the most serious faults detected today?" # Perform semantic search (top 5 most similar)
results = vec.search(query_text=query,
                     limit=5) # Show results clearly
for i,
    row in results.iterrows(): print(f"\n--- Result {i + 1} ---") print(f"Asset: {row.get('asset', 'N/A')}") print(f"Fault ID: {row.get('fault_id', 'N/A')}") print(f"Fault Level: {row.get('fault_lvl', 'N/A')}") print(f"Detected: {row.get('created_at', 'N/A')}") print(f"Distance: {row.get('distance', 'N/A'):.4f}") print("\nSummary:") print(row["contents"])