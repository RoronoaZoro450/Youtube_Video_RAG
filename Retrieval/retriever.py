from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_core.runnables import RunnableLambda

load_dotenv()


def hybrid_search(vector_store, docs,k):

    bm25_retriever = BM25Retriever.from_documents(docs,k=k) # Adjust k as needed
    retriever = vector_store.as_retriever(search_kwargs={"k": k}) # Adjust k as needed

    ensemble_retriever = EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.5, 0.5])
    
    return ensemble_retriever

def hybrid_search_node(inputs):
    ensemble_retriever = hybrid_search(
        inputs["vector_store"],
        inputs["docs"],
        inputs["k"],
    )
    return {
        **inputs,
        "ensemble_retriever": ensemble_retriever
    }

hybrid_search_runnable = RunnableLambda(hybrid_search_node)

def create_multi_query_retriever(ensemble_retriever, RAG_llm):
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=RAG_llm
    )
    return multi_query_retriever

def multi_query_retriever_node(inputs):
    multi_retriever = create_multi_query_retriever(
        inputs["ensemble_retriever"],
        inputs["RAG_llm"]
    )
    return {
        **inputs,
        "multi_query_retriever": multi_retriever
    }

multi_query_retriever_runnable = RunnableLambda(multi_query_retriever_node)


def retrieve_docs_node(inputs):

    retrieved_docs = inputs["multi_query_retriever"].invoke(
        inputs["enhanced_query"]
    )

    return {
        **inputs,
        "retrieved_docs": retrieved_docs
    }

retrieve_docs_runnable = RunnableLambda(
    retrieve_docs_node
)

def deduplicate_docs(retrieved_docs):
    seen = set()
    deduped_docs = []
    for doc in retrieved_docs:
        if doc.page_content not in seen:
            deduped_docs.append(doc)
            seen.add(doc.page_content)
    return deduped_docs


def deduplicate_docs_node(inputs):
    
    deduped_docs = deduplicate_docs(inputs["retrieved_docs"])

    return {
        **inputs,
        "deduped_docs": deduped_docs
    }

deduplicate_docs_runnable = RunnableLambda(deduplicate_docs_node)