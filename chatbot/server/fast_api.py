from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from travel_agent import TravelAgent

from pathlib import Path

HOME = Path.home()

config = {
    "model_path": str(HOME / "Github/smart-travel/chatbot/models/mistral-finetuned-v0.2"),
    "base_model_path": str(HOME / "Github/smart-travel/chatbot/models/mistral"),
}

print("Loading the models")
travel_agent = TravelAgent(config=config)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request model for API
class QueryRequest(BaseModel):
    prompt: str
    stream: bool

@app.post("/chat/v1")
async def submit_query(request: QueryRequest):
    response = travel_agent.generate(request.prompt, stream=request.stream)

    if request.stream:
        return StreamingResponse(response, media_type="text/event-stream")

    return {"query": request.prompt, "response": response}

@app.post("/chat/base")
async def submit_query(request: QueryRequest):
    response = travel_agent.generate_base(request.prompt, stream=request.stream)

    if request.stream:
        return StreamingResponse(response, media_type="text/event-stream")

    return {"query": request.prompt, "response": response}

