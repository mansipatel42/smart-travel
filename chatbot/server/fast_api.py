from fastapi import FastAPI
from pydantic import BaseModel
from travel_agent import TravelAgent

config = {
    "model_path": "C:/Users/vedmp/smart-travel/fine-tuning/models/mistral-finetuned",
    "base_model_path": "C:/Users/vedmp/smart-travel/fine-tuning/models/mistral",
}

print("Loading the model")
travel_agent = TravelAgent(config=config)

app = FastAPI()

# Request model for API
class QueryRequest(BaseModel):
    query: str

@app.post("/chat/v1")
async def submit_query(request: QueryRequest):
    response = travel_agent.generate(request.query)
    return {"query": request.query, "response": response}

@app.post("/chat/v0")
async def submit_query(request: QueryRequest):
    response = travel_agent.generate_base(request.query)
    return {"query": request.query, "response": response}