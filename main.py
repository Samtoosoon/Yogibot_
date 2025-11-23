from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from logic.llm import YogiLLM
from logic.rag import get_rag_engine
import os
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
SYSTEM_API_KEY = os.getenv("gemini_api_key")

if not SYSTEM_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")

app = FastAPI()

# Input Models (No apiKey required from client anymore)
class ChatRequest(BaseModel):
    mode: str
    language: str
    query: str
    code: str = ""
    error: str = ""

class SummarizeRequest(BaseModel):
    text: str

# 2. Serve the Frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

# 3. API Endpoint: Chat
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not SYSTEM_API_KEY:
        raise HTTPException(status_code=500, detail="Server Configuration Error: API Key missing.")
    
    try:
        # Init Logic Layers with System Key
        rag = get_rag_engine()
        llm = YogiLLM(SYSTEM_API_KEY)
        
        # RAG Search
        context_docs = rag.search(req.query + " " + req.language)
        
        # LLM Generation
        response = llm.generate_code_response(
            req.mode, req.query, req.code, req.error, context_docs
        )
        
        return {
            "response": response,
            "context": [d['source'] for d in context_docs]
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. API Endpoint: Summarize
@app.post("/api/summarize")
async def summarize_endpoint(req: SummarizeRequest):
    if not SYSTEM_API_KEY:
        raise HTTPException(status_code=500, detail="Server Configuration Error: API Key missing.")

    try:
        llm = YogiLLM(SYSTEM_API_KEY)
        summary = llm.summarize_article(req.text)
        return {"response": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)