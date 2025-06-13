import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)

model = "gemini-2.0-flash-001"
prompt = "why Gemini has so difficult documentation?"

def generate_text(prompt):
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response

if __name__ == "__main__":
    response = generate_text(prompt)
    print(response.text)
    print("Prompt tokens:", response.usage_metadata.prompt_token_count)
    print("Response tokens:", response.usage_metadata.candidates_token_count)
