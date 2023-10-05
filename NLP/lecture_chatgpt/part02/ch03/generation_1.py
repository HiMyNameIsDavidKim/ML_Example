import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

file = open(r"/Users/davidkim/security/openai.txt", "r", encoding='UTF8')
data = file.read()
KEY_NLP = str(data)
file.close()

openai.api_key = KEY_NLP

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    temperature: float = 1


SYSTEM_MSG = "You are a helpful travel assistant, Your name is Jini, 27 years old"

@app.get("/")
async def root():
    return {"message": "Hello FastAPI!"}

@app.post("/chat")
def chat(req: ChatRequest):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": req.message},
        ],
        temperature=req.temperature,
    )
    return {"message": response.choices[0].message.content}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
