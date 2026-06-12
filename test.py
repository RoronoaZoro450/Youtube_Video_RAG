from supadata import Supadata, SupadataError
from dotenv import load_dotenv
from supadata.types import BatchJob
import os

load_dotenv()  # Load environment variables from .env file

# Initialize the client
supadata = Supadata(api_key=os.getenv("SUPADATA_API_KEY"))


transcript = supadata.transcript(
    url="https://www.youtube.com/watch?v=8rO0cBPLfb0",
    text=True,
    mode="auto"
)

print(type(transcript.content))