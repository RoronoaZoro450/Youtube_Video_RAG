from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

str_output_parser = StrOutputParser()

def answer_llm(query, context, RAG_llm, chat_history):
    chat_history = chat_history or []

    context_str = "\n\n".join(
        [doc.page_content for doc in context]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question-answering assistant for a YouTube video RAG pipeline.
        Answer the question using ONLY the provided context from the video transcript.

        STRICT RULES:
        - If the answer is not in the context, say exactly: "This video does not cover that."
        - If partially covered, answer what is available, then say: "The video does not fully cover this topic."
        - Never guess, never add outside knowledge, never hallucinate
        - If you don't know, say: "I don't know, it is not covered in the video."
        - Keep answers concise and direct
        - You may use conversation history ONLY to understand what "it", "that", "this" refers to — never to answer from history instead of context
        - When listing items, only list what the video explicitly frames as a distinct point
        - Do not convert tips, examples, or habits into separate list items
        - If the video gives a clear hierarchy (main skill vs how to practice it), preserve that hierarchy
        Context from video transcript:
        {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])

    answer_chain = prompt | RAG_llm | str_output_parser
    response = answer_chain.invoke({
        "context": context_str,
        "chat_history": chat_history,
        "query": query
    })

    return response

def answer_llm_node(inputs):
    answer = answer_llm(
        inputs["query"],
        inputs["reranked_docs"],
        inputs["RAG_llm"],
        inputs["chat_history"]
    )

    return {
        **inputs,
        "answer": answer
    }

answer_llm_runnable = RunnableLambda(
    answer_llm_node
)
