from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Initialize the model (e.g., gemini-2.5-flash or gemini-2.5-pro)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Invoke the model directly
response = llm.invoke("Explain LangChain in one sentence.")
print(response.content)

"""
for creative thinking and reasoning, you can use the "gemini-3.1-pro-preview" model with the following configuration:

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    thinking_level="medium",  # Controls the depth of reasoning (low, medium, high)
    include_thoughts=True     # Includes the model's logic flow in the payload
)


For more structured interactions, you can use the ChatPromptTemplate to define a conversation flow:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sarcastic coding assistant. Keep your answers brief."),
    ("user", "How do I fix a NullPointerException in Java?")
])

chain = prompt | llm

"""