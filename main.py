from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from Pipeline.query_chain import run_rag_chain
from Ingestion.ingestion_chain import run_ingestion_chain
import shutil
from Pipeline.chat_memory import update_chat_history, get_chat_history, clear_chat_history
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


def main(url):
    
    RAG_llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="conversational",
        temperature=0.1,)
    RAG_llm = ChatHuggingFace(llm=RAG_llm_endpoint)

    print("Hello from Youtube-RAG!")

    # url = "https://www.youtube.com/watch?v=gdgZ-X87Bwg" # Replace with your desired YouTube video URL
    url = input("Enter the YouTube video URL: ")

    result = run_ingestion_chain(url)

    docs = result["documents"]
    vector_store = result["vector_store"]

    
    while True: 
        query = input("Enter your query (or 'stop' to exit): ") # Replace with your desired query
        if query == "stop":
            shutil.rmtree("./chroma_db", ignore_errors=True)
            print("Stopping the RAG chain.")
            clear_chat_history()
            break

        llm_res=run_rag_chain(query, docs, RAG_llm, vector_store, k=10, top_n=8, chat_history=get_chat_history())

        update_chat_history(query, llm_res["answer"])

        print("Final Answer:")
        print(llm_res["answer"])


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=gdgZ-X87Bwg"
    main(url)
    
