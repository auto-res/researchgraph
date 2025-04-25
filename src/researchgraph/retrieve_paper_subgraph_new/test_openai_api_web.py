from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    tools=[{"type": "web_search_preview"}],
    input="Please provide a comprehensive report on recent advancements in optimization techniques, with a particular focus on improvements to the Adam optimizer. The report should reference up-to-date research from leading conferences such as NeurIPS and ICLR."
)

print(response.output_text)
