from litellm import acompletion
import asyncio
from pydantic import BaseModel
import ast


class QueryInfo(BaseModel):
    query: str
    research_goal: str


class QueryInfoList(BaseModel):
    queries_list: list[QueryInfo]


async def generate_queries(
    llm_name: str, query: str, num_queries: int = 3, learnings: list[str] = None
) -> list[str]:
    """
    ユーザーの質問から検索クエリを生成する。

    Args:
        query (str): ユーザーの質問
        num_queries (int, optional): 生成する検索クエリの数 (デフォルト: 3)
        learnings (List[str], optional): 過去の学習内容 (オプション)

    Returns:
        List[Dict[str, str]]: 検索クエリのリスト (query, research_goal)
    """
    system_prompt = "You are an expert research assistant."

    prompt_text = f"""
Given the following user query, generate up to {num_queries} unique SERP queries to research the topic.
Ensure each query is distinct and not redundant.
User Query: {query}"""

    if learnings:
        prompt_text += "Here are some insights from previous research. Use these to refine the new queries:\n"
        prompt_text += "\n".join(learnings) + "\n\n"

    prompt_text += "Return the queries in JSON format as a list of objects with keys 'query' and 'research_goal'."

    response = await acompletion(
        model=llm_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        response_format=QueryInfoList,
        temperature=0.9,
    )
    output = response.choices[0].message.content
    output_dict = ast.literal_eval(output)
    # queries_list = output_dict["queries_list"]
    queries_list = QueryInfoList(**output_dict)
    return queries_list  # [:num_queries]


async def main():
    llm_name = "gpt-4o-mini-2024-07-18"
    query = "What are the key concepts in machine learning?"
    num_queries = 3
    output = await generate_queries(
        llm_name=llm_name,
        query=query,
        num_queries=num_queries,
    )
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
