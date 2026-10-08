import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found.")

print("API key found.")

http_client = httpx.Client(verify=False)

client = OpenAI(
    api_key=api_key,
    http_client=http_client
)

response = client.responses.create(
    model="gpt-5.6-luna",
    input="Reply with exactly: OpenAI API connection successful."
)

print(response.output_text)