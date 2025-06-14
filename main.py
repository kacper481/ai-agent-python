import os
from dotenv import load_dotenv
import sys

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)

model = "gemini-2.0-flash-001"

if len(sys.argv) > 1:
    user_prompt = sys.argv[1]
else:
    print("Usage: python main.py <model_name>")
    sys.exit(1)

messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

def generate_answer(user_prompt):
    response = client.models.generate_content(
        model=model,
        contents=messages,
    )
    return response

def print_metadata(response):
    print("User prompt:", user_prompt)
    print("Prompt tokens:", response.usage_metadata.prompt_token_count)
    print("Response tokens:", response.usage_metadata.candidates_token_count)

if __name__ == "__main__":
    response = generate_answer(user_prompt)
    print(response.text)
    if "--verbose" in sys.argv:
        print_metadata(response)
