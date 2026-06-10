import shutil
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from Pipeline.query_chain import run_rag_chain
from Ingestion.ingestion_chain import run_ingestion_chain
from Pipeline.chat_memory import get_chat_history, update_chat_history, clear_chat_history
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# pydantic models for request bodies

class IngestRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

# Global state to manage the API's current status and resources

Api_States = {
    "llm": None,
    "vector_store": None,
    "ingestion": False,
    "docs": None,
    "query": False
}

# FAST API app with lifespan management for resource initialization and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    RAG_llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="conversational",
        temperature=0.1,)
    Api_States["llm"] = ChatHuggingFace(llm=RAG_llm_endpoint)

    yield
    
    # Cleanup code if needed
    Api_States["llm"] = None
    Api_States["vector_store"] = None
    Api_States["ingestion"] = False
    Api_States["docs"] = None
    Api_States["query"] = False
    shutil.rmtree("./chroma_db", ignore_errors=True)
    clear_chat_history()

app = FastAPI(lifespan=lifespan)


@app.get('/')
def read_root():
    return {"message": "Welcome to the YouTube RAG API!"}


@app.post('/ingest_video')
def ingest_video(request: IngestRequest):
    try:
        result = run_ingestion_chain(request.url)
        Api_States["vector_store"] = result["vector_store"]
        Api_States["docs"] = result["documents"]
        Api_States["ingestion"] = True
        return { "message": "Video ingested successfully!" }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/query')
def query_video(request: QueryRequest):
    if not Api_States["ingestion"]:
        raise HTTPException(status_code=400, detail="Please ingest a video first.")

    try:
        llm_res = run_rag_chain(
            request.query,
            Api_States["docs"],
            Api_States["llm"],
            Api_States["vector_store"],
            k=10,
            top_n=8,
            chat_history=get_chat_history(),
        )
        Api_States["query"] = True
        update_chat_history(request.query, llm_res["answer"])
        return { "answer": llm_res["answer"] }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/clear_history')
def clear_history():
    clear_chat_history()
    return { "message": "Chat history cleared successfully!" }

@app.get('/chat_history')
def route_chat_history():
    return { "history": get_chat_history() }

@app.post('/reset')
def reset():
    shutil.rmtree("./chroma_db", ignore_errors=True)
    clear_chat_history()
    Api_States["vector_store"] = None
    Api_States["docs"] = None
    Api_States["ingestion"] = False
    Api_States["query"] = False
    return {"message": "Session reset."}