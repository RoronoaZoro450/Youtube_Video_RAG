from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langsmith import traceable
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_core.runnables import RunnableLambda

load_dotenv()

# embeddings = HuggingFaceEndpointEmbeddings(
#     model="BAAI/bge-small-en-v1.5"
# )

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
)

@traceable(name="create_vector_store", project_name="YoutubeRAG")
def create_vector_store(docs):

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return db

def create_vector_store_node(inputs):
    vector_store = create_vector_store(inputs["documents"])
    return {
        **inputs,
        "vector_store": vector_store
    }

create_vector_store_runnable = RunnableLambda(create_vector_store_node)
