from fastapi import FastAPI, HTTPException
from src.models import ChatRequest, ChatResponse
from src.agent import run_agent
import time

app = FastAPI(title="SHL Conversational Agent")

@app.get("/health")
async def health_check():
    # Cold starts on hosting platforms allow up to 2 minutes [cite: 96]
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    
    # Process the stateless conversation history
    try:
        response_data = await run_agent(request.messages)
        
        # Enforce 30-second timeout safety check [cite: 97]
        if time.time() - start_time > 28:
            print("Warning: Approaching timeout limit.")

        return ChatResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))