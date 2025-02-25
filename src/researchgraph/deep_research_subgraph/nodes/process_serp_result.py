from litellm import acompletion
import asyncio
from pydantic import BaseModel
import json

from researchgraph.deep_research_subgraph.nodes.request_firecrawl_api import (
    SearchResponse,
    SearchResponseItem,
)


class ResearchResult(BaseModel):
    learnings: list[str]
    followup_questions: list[str]


async def process_serp_result(
    llm_name: str,
    query: str,
    result: SearchResponse,
    num_learnings: int = 3,
    num_followup_questions: int = 3,
) -> ResearchResult:
    contents = [
        await trim_prompt(item.markdown, 25000) for item in result if item.markdown
    ]
    output = await generate_object(
        llm_name, query, contents, num_learnings, num_followup_questions
    )
    processed_result = ResearchResult(**output)
    return processed_result


async def trim_prompt(content: str, max_length: int) -> str:
    """コンテンツの長さを制限する関数"""
    return content[:max_length]


async def generate_object(
    llm_name: str,
    query: str,
    contents: list[str],
    num_learnings: int,
    num_followups: int,
) -> tuple[list[str], list[str]]:
    prompt = f"""Given the following contents from a SERP search for the query <query>{query}</query>, generate a list of learnings from the contents.
    Return a maximum of {num_learnings} learnings, but feel free to return less if the contents are clear.
    Make sure each learning is unique and not similar to each other.
    The learnings should be concise and to the point, as detailed and information-dense as possible.
    Include entities like people, places, companies, products, things, etc., as well as any exact metrics, numbers, or dates.
    The learnings will be used to research the topic further.

    <contents>
    {''.join(f"<content>{c}</content>" for c in contents)}
    </contents>
    
    # Output format:
    List of learnings, max of {num_learnings}
    List of follow-up questions to research the topic further, max of {num_followups}
    """

    response = await acompletion(
        model=llm_name,
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format=ResearchResult,
    )
    output = response.choices[0].message.content
    output_dict = json.loads(output)
    learnings = output_dict["learnings"]
    followup_questions = output_dict["followup_questions"]
    return {"learnings": learnings, "followup_questions": followup_questions}


async def main():
    llm_name = "gpt-4o-mini-2024-07-18"
    sample_result = SearchResponse(
        search_data=[
            SearchResponseItem(
                markdown="This is some example content from the search result.",
                url="https://example.com",
            )
        ]
    )
    query = "Example Query"
    processed_result = await process_serp_result(llm_name, query, sample_result)
    print(processed_result.learnings)
    print(processed_result.followup_questions)


if __name__ == "__main__":
    asyncio.run(main())
