from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda



splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[". ", "? ", "! ", ", ", " "]
)

def create_documents(chunks):
    docs = []

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()

        if chunk:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"chunk_id": i}
                )
            )

    return docs


def split_text(text):                       # main function to split text into chunks and create documents
    chunks = splitter.split_text(text)
    documents = create_documents(chunks)
    return documents

def split_text_node(inputs):
    docs = split_text(
        inputs["transcript"]
        )
    return {
        **inputs,
        "documents": docs
    }

split_text_runnable = RunnableLambda(split_text_node)
