from openai import OpenAI


def openai_client(model_name: str, prompt: str) -> str:
    client = OpenAI()

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "developer",
                "content": "You are an experienced researcher in machine learning, and you are well versed in theory and can also implement it well.",
            },
            {"role": "user", "content": f"{prompt}"},
        ],
    )

    output = completion.choices[0].message.content
    return output
