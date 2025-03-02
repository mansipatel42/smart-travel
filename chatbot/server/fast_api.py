import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from travel_agent import TravelAgent
from queue import Queue
from time import sleep
from threading import Thread

config = {
    "model_path": "C:/Users/vedmp/smart-travel/fine-tuning/models/mistral-finetuned-v0.2",
    "base_model_path": "C:/Users/vedmp/smart-travel/fine-tuning/models/mistral",
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

"""
@app.get('/chat/stream')
async def stream():
    return StreamingResponse(data_reader(), media_type='text/event-stream')
"""