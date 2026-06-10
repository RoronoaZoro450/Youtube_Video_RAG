from Ingestion.yt_context_loader import fetch_transcript_runnable
from Ingestion.context_chunking import split_text_runnable
from Retrieval.vector_store import create_vector_store_runnable

Ingestion_chain = fetch_transcript_runnable | split_text_runnable | create_vector_store_runnable

def run_ingestion_chain(url):
    result = Ingestion_chain.invoke({"url": url})
    return result
