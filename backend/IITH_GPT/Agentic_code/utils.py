import requests
from llama_index.core.llms import ChatMessage, MessageRole
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Moves up to root
load_dotenv(BASE_DIR / ".env")

# Define Gemini API endpoint and API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def classify_query_with_gemini(query):
    """
    Classify a user query as 'summarization' or 'question_answering' using Google Gemini API.
    """
    # Construct the API URL with the key
    api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    # Create the request payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"""
                        Classify the following query into one of these types:
                        - 'summarization'
                        - 'question_answering'
                        - 'search'
                        - 'fact_verification'
                        - 'exploration'

                        Query: {query}

                        Examples:
                        1. What is the capital of India?
                           Output: question_answering
                        2. Summarize the given paragraph.
                           Output: summarization
                        3. Find documents on climate change policies.
                           Output: search
                        4. Verify if the claim 'Earth is flat' is true.
                           Output: fact_verification
                        5. Explore the history of space exploration.
                           Output: exploration
                        """
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Make a POST request to the Gemini API
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # Extract the classification result
        classification = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip().lower()
        # print(f"Predicted class: {classification}")
        return classification

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        return "error"
    

def chunk_text(text, chunk_size):
    """Split text into smaller chunks for parallel processing."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize_history(summarizer, messages):
    history_text = " ".join([msg.content for msg in messages if msg.role != MessageRole.SYSTEM])

    # No need to split if text is short
    if len(history_text) <= 1000:
        return summarizer(history_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    # Handle longer texts by splitting
    chunks = chunk_text(history_text, chunk_size=1000)  # Use an optimized chunk size
    summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)