from fastapi import FastAPI, HTTPException
from src.models import ChatRequest, ChatResponse
from src.agent import run_agent
import time

app = FastAPI(title="SHL Conversational Agent")


@app.get("/health")
async def health_check():
    """
    Lightweight health check — intentionally does NOT load the model.
    Railway calls this to decide if the container is alive. Keeping it
    instant lets the service pass the check even before the first real
    request triggers the lazy model load.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    try:
        response_data = await run_agent(request.messages)

        elapsed = time.time() - start_time
        if elapsed > 28:
            print(f"Warning: Response took {elapsed:.1f}s — approaching timeout.")

        return ChatResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))