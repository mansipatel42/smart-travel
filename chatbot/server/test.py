# from travel_agent import TravelAgent
# from pathlib import Path
# 
# HOME = Path.home()
# 
# config = {
#     "model_path": str(HOME / "Github/smart-travel/chatbot/models/mistral-finetuned-v0.2"),
#     "base_model_path": str(HOME / "Github/smart-travel/chatbot/models/mistral"),
# }
# 
# agent = TravelAgent(config=config)
# 
# agent.generate(
#     prompt="I am planning a trip to Chigaco. Can you create a 2 day itenary for me?",
#     stream=True
# )
import requests

url = "http://127.0.0.1:8000/chat/base"
data = {
    "prompt": "I am planning a trip to Chigaco. Can you create a 2 day itenary for me?",
    "stream": True
}

# Create a request object
req = requests.Request("POST", url, json=data)

# Prepare the request
prepared = req.prepare()

# Function to print the prepared request details
def print_request(prepared):
    print("========== REQUEST DETAILS ==========")
    print(f"URL: {prepared.url}")
    print(f"Method: {prepared.method}")
    print("Headers:")
    for k, v in prepared.headers.items():
        print(f"  {k}: {v}")
    print("\nBody:")
    if prepared.body:
        print(prepared.body.decode() if isinstance(prepared.body, bytes) else prepared.body)
    print("=====================================")

# Print request details
print_request(prepared)

# Send the request using a session
with requests.Session() as session:
    with session.send(prepared, stream=True) as r:
        for chunk in r.iter_content(1024):
            print(chunk.decode("utf-8"), sep="", end="")
