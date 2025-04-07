import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_PATH

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = db.get_or_create_collection(name="knowledge_base")

# Sample documents
documents = [
    "Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving external knowledge.",
    "CUDA acceleration improves AI model performance by parallelizing computations.",
    "Qwen2.5-3B-GPTQ is optimized for retrieval-augmented generation."
]

# Store documents in ChromaDB
embeddings = embedding_model.encode(documents)
for doc, emb in zip(documents, embeddings):
    collection.add(ids=[str(hash(doc))], documents=[doc], embeddings=[emb.tolist()])

print("✅ Knowledge Base Initialized")
