from openai import OpenAI
client = OpenAI()

# Print API key availability
import os
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print("OpenAI API key found")
else:
    print("WARNING: OpenAI API key not found in environment variables")

# Simple test without web search capability
print("Testing basic API functionality without web search...")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What is the Adam optimizer in machine learning? Keep it brief."}
    ]
)

print(response.choices[0].message.content)
