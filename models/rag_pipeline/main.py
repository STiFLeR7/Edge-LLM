from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import retrieve_context
from generator import generate_response

app = FastAPI()

# ✅ Allow frontend to send requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],  # ✅ Allows all headers
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")  # Root endpoint to check if FastAPI is running
def root():
    return {"message": "FastAPI is running!"}

@app.post("/query/")
async def query_rag(request: QueryRequest):
    """Retrieves context and generates a response."""
    context = retrieve_context(request.query)
    prompt = f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:"
    return {"response": generate_response(prompt)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2030)
