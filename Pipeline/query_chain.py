from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from Pipeline.query_enhance import query_enhance_runnable
from Retrieval.retriever import multi_query_retriever_runnable, deduplicate_docs_runnable, hybrid_search_runnable ,retrieve_docs_runnable
from Retrieval.cohere_reranker import cohere_reranker_runnable
from Pipeline.llm_response import answer_llm_runnable

rag_chain=  ( query_enhance_runnable
        | hybrid_search_runnable
        | multi_query_retriever_runnable
        | retrieve_docs_runnable
        | deduplicate_docs_runnable 
        | cohere_reranker_runnable
        | answer_llm_runnable
        )



def run_rag_chain(query, docs, RAG_llm, vector_store, k=10, top_n=8, chat_history=None):
    final_output= rag_chain.invoke({
            "query": query,
            "docs": docs,
            "RAG_llm": RAG_llm,
            "k": k,
            "top_n": top_n,
            "vector_store": vector_store,   
            "chat_history": chat_history 
        })

    return final_output
