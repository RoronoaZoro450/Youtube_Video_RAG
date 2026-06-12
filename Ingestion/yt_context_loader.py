from langchain_core.runnables import RunnableLambda
from supadata import Supadata, SupadataError
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# Initialize the client
supadata = Supadata(api_key=os.getenv("SUPADATA_API_KEY"))

def fetch_transcript(url):
    transcript = supadata.transcript(
        url=url,
        text=True,
        mode="auto"
    )
    return transcript.content


def fetch_transcript_runnable(inputs):
    url = fetch_transcript(
        inputs["url"]
        ) 
    return {
        **inputs,
        "transcript": url
    }

fetch_transcript_runnable = RunnableLambda(fetch_transcript_runnable)
