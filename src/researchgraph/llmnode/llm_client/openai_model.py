from openai import OpenAI

client = OpenAI()


def opnai_structured_output(llm_name: str, prompt: str, response_field) -> dict:
    completion = client.beta.chat.completions.parse(
        model=llm_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=response_field,
    )
    event = completion.choices[0].message.parsed
    event = event.dict()
    return event
