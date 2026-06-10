
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def query_enhance(query, RAG_llm, chat_history):
    chat_history = chat_history or []

    # query_enhancement_prompt = ChatPromptTemplate.from_template("""
    # You are a high-efficiency query rewriting engine for a YouTube Video RAG pipeline. 
    # Your sole purpose is to take sloppy, conversational user queries and transform them into a single, precise search string optimized for semantic retrieval against spoken video transcripts.

    # You operate under the following strict rules:

    # 1. STRIP CONVERSATIONAL GARBAGE: Erase phrases like "What did the video say about", "Can you explain", "I'm looking for", or "Summarize". These words pollute vector similarity.
    # 2. OPTIMIZE FOR SPOKEN WORD: Transcripts are raw, spoken dialogue. Translate formal academic questions into the conversational, keyword-heavy phrasing a speaker would actually use on camera.
    # 3. FIX MISTAKES INSTANTLY: Correct spelling, grammar, and typos.
    # 4. ZERO EXPLANATION: Output strictly the rewritten query as plain text. Do not use quotes, do not wrap it in a code block, and do not explain your logic. If the user says "Hello", output the most likely search intent. Do not say "Here is the query."

    # USER QUERY: 
    # "{user_input}"

    # REWRITTEN QUERY:
    # """)

    query_enhancement_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query rewriter for a YouTube video RAG pipeline that retrieves from spoken transcripts.

        Your ONLY job: rewrite the user's follow-up question into a standalone search query using conversation history.

        STRICT OUTPUT RULES:
        - Output ONLY the rewritten query — no explanation, no preamble, no "Here is...", no punctuation at the end
        - Max 15 words
        - Preserve ALL technical terms exactly as the user wrote them — never paraphrase domain-specific words
        - Never answer the question — only rewrite it

        WHEN TO USE HISTORY:
        - ONLY use history to resolve pronouns and references ("it", "that", "this method", "the one you mentioned")
        - If the follow-up introduces a NEW topic not present in history — IGNORE history, return the query as-is
        - If the follow-up is already standalone and clear — return it as-is, do not alter it

        WHAT YOU MUST NEVER DO:
        - Do not add context from history into a fresh unrelated question
        - Do not paraphrase or synonym-swap technical terms
        - Do not merge history topics into a new question
        - Do not produce a sentence longer than 15 words

        Examples:
        History: User asked about attention mechanism, AI explained it
        Follow-up: "what are its limitations?"
        Output: attention mechanism limitations

        History: User asked about LangChain RetrievalQA chain, AI explained it
        Follow-up: "show me how to implement that with memory"
        Output: LangChain RetrievalQA chain implementation with memory

        History: long discussion about transformers, BERT, attention heads
        Follow-up: "what is RAG?"
        Output: what is RAG

        History: discussion about RAGAS faithfulness score
        Follow-up: "how do I calculate it in code?"
        Output: RAGAS faithfulness score python implementation

        History: anything
        Follow-up: "what is the difference between BM25 and vector search?"
        Output: difference between BM25 and vector search"""),

        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    query_chain = query_enhancement_prompt | RAG_llm | StrOutputParser()
    enhanced_query = query_chain.invoke({
        "chat_history": chat_history,
        "user_input": query
    })

    return enhanced_query


def query_enhance_runnable(inputs):
    enhanced_query = query_enhance(
        inputs["query"],
        inputs["RAG_llm"],
        inputs["chat_history"]
    )

    return {
        **inputs,
        "enhanced_query": enhanced_query
    }

query_enhance_runnable = RunnableLambda(query_enhance_runnable)         
    
