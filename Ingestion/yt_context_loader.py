
from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.parse import urlparse, parse_qs 
from langchain_core.runnables import RunnableLambda

ytt_api = YouTubeTranscriptApi() 

# Fetches the transcript for a given YouTube video URL, cleans it, and returns the cleaned text.


def clean_transcript(text):

    # Remove weird unicode spaces
    text = text.replace('\xa0', ' ')

    # Remove repeated spaces/newlines
    text = re.sub(r'\s+', ' ', text)

    # Remove filler speech words
    fillers = [
        r'\buh\b',
        r'\bum\b',
        r'\byou know\b',
        r'\blike\b',
        r'\bkind of\b',
        r'\bsort of\b'
    ]

    for filler in fillers:
        text = re.sub(filler, '', text, flags=re.IGNORECASE)

    # Remove excessive punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    # Normalize spaces again
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def fetch_transcript(URL): # Main Function

    parsed_url = urlparse(URL)
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]

    transcript = ytt_api.fetch(video_id, languages=["hi", "en"])

    full_text = " ".join([entry.text for entry in transcript])
    clean_text = re.sub(r" >> ", "", full_text)

    clean_text = clean_transcript(clean_text)
    # print(clean_text)
    return clean_text

def fetch_transcript_runnable(inputs):
    url = fetch_transcript(
        inputs["url"]
        ) 
    return {
        **inputs,
        "transcript": url
    }

fetch_transcript_runnable = RunnableLambda(fetch_transcript_runnable)
