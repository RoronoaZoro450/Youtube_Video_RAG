import cohere
from langchain_core.runnables import RunnableLambda

def cohere_reranker(new_query, deduped_retrieved_docs, top_n):
    if not deduped_retrieved_docs:
        return []

    reranking_model = cohere.ClientV2()
    response = reranking_model.rerank(
        model="rerank-v3.5",
        query=new_query,
        documents=[doc.page_content for doc in deduped_retrieved_docs],
        top_n=top_n
    )

    reranked_docs = []

    for result in response.results:
        doc = deduped_retrieved_docs[result.index]
        doc.metadata["rerank_score"] = result.relevance_score
        reranked_docs.append(doc)

    return reranked_docs

def cohere_reranker_node(inputs):
    reranked_docs = cohere_reranker(
        inputs["enhanced_query"],
        inputs["deduped_docs"],
        inputs["top_n"]
    )

    return {
        **inputs,
        "reranked_docs": reranked_docs
    }

cohere_reranker_runnable = RunnableLambda(cohere_reranker_node)
