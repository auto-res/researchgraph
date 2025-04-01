from researchgraph.utils.openai_client import openai_client


def generator_node(
    llm_name: str,
    base_method_text: str,
    add_method_text_list: list[str],
) -> str:
    add_method_text = "---".join(add_method_text_list)
    prompt = f"""
Your task is to propose a genuinely novel method that mitigates one or more challenges in the “Base Method.” This must go beyond a mere partial modification of the Base Method. To achieve this, reason through the following steps and provide the outcome of step 3:
- Identify multiple potential issues with the “Base Method.”
- From the “Add Method” approaches, select one that can address at least one of the issues identified in step 1.
- Drawing inspiration from both the “Base Method” and the selected approach in step 2, devise a truly new method.

# Base Method
{base_method_text}

# Add Method
{add_method_text}
"""
    messages = [
        {"role": "user", "content": f"{prompt}"},
    ]
    response = openai_client(model_name=llm_name, message=messages)
    return response
