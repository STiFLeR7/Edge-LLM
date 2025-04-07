import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_PATH
import time  # ✅ Added for measuring retrieval time

# ✅ Switch to a faster embedding model
embedding_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# Load ChromaDB
db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = db.get_or_create_collection(name="knowledge_base")

def retrieve_context(query, top_k=1):  # ✅ Reduced top_k to speed up retrieval
    """Retrieve relevant documents from ChromaDB based on the query."""
    start_time = time.time()  # Start time measurement
    query_embedding = embedding_model.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
    
    retrieval_time = time.time() - start_time  # ✅ Measure retrieval time
    print(f"Retrieval Time: {retrieval_time:.4f}s")  

    if results["documents"]:
        return "\n".join(results["documents"][0])
    return "No relevant documents found."
