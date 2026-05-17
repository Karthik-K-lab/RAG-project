from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rag_pipeline

app = FastAPI()   

origins = ["http://localhost:5173", "http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,

    allow_origins=origins,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

class Query(BaseModel):
    question : str
    context : str


@app.get('/')
def home():
    return {"message" : "Home"}

@app.post('/ask')
async def take_input(query: Query):

    query_input, text = query.question, query.context

    response, top_chunks = rag_pipeline.pipeline(query_input, text)
    
    return {
        "query" : query.question,
        "answer" : response,
        "top_chunks" : top_chunks
        }