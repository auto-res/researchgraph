from litellm import acompletion
import asyncio
from pydantic import BaseModel
import json

from researchgraph.deep_research_subgraph.nodes.request_firecrawl_api import (
    SearchResponseItem,
)


class ResearchResult(BaseModel):
    learnings: list[str]
    followup_questions: list[str]


async def process_serp_result(
    llm_name: str,
    query: str,
    result: list[SearchResponseItem],
    num_learnings: int = 3,
    num_followup_questions: int = 3,
) -> ResearchResult:
    """
    検索結果を処理して学習内容とフォローアップ質問を抽出する

    Args:
        llm_name: 使用するLLMの名前
        query: 検索クエリ
        result: 検索結果のリスト
        num_learnings: 抽出する学習内容の最大数
        num_followup_questions: 生成するフォローアップ質問の最大数

    Returns:
        学習内容とフォローアップ質問を含むResearchResult
    """
    print(f"  Processing {len(result)} search results to extract learnings and follow-up questions")

    # 検索結果が空の場合はデフォルト値を返す
    if not result:
        print("  Warning: No search results to process")
        return ResearchResult(
            learnings=["No information found for the given query."],
            followup_questions=[
                f"What are the basics of {query}?",
                f"Why is {query} important?",
                f"What are the latest developments in {query}?"
            ]
        )

    # 各検索結果のマークダウンコンテンツを取得し、長さを制限
    contents = []
    for item in result:
        if item.markdown:
            trimmed_content = await trim_prompt(item.markdown, 25000)
            if trimmed_content:
                contents.append(trimmed_content)

    # コンテンツが空の場合はデフォルト値を返す
    if not contents:
        print("  Warning: No content found in search results")
        return ResearchResult(
            learnings=["No detailed content found for the given query."],
            followup_questions=[
                f"What are the basics of {query}?",
                f"Why is {query} important?",
                f"What are the latest developments in {query}?"
            ]
        )

    # LLMを使用して学習内容とフォローアップ質問を生成
    print(f"  Generating learnings and follow-up questions from {len(contents)} content items")
    output = await generate_object(
        llm_name, query, contents, num_learnings, num_followup_questions
    )

    processed_result = ResearchResult(**output)
    print(f"  Generated {len(processed_result.learnings)} learnings and {len(processed_result.followup_questions)} follow-up questions")

    return processed_result


async def trim_prompt(content: str, max_length: int) -> str:
    """
    コンテンツの長さを制限する関数

    Args:
        content: 制限するコンテンツ
        max_length: 最大長

    Returns:
        制限されたコンテンツ
    """
    if len(content) > max_length:
        print(f"  Trimming content from {len(content)} to {max_length} characters")
    return content[:max_length]


async def generate_object(
    llm_name: str,
    query: str,
    contents: list[str],
    num_learnings: int,
    num_followups: int,
) -> dict:
    """
    LLMを使用して検索結果から学習内容とフォローアップ質問を生成する

    Args:
        llm_name: 使用するLLMの名前
        query: 検索クエリ
        contents: 検索結果のコンテンツリスト
        num_learnings: 抽出する学習内容の最大数
        num_followups: 生成するフォローアップ質問の最大数

    Returns:
        学習内容とフォローアップ質問を含む辞書
    """
    # プロンプトの作成
    prompt = f"""Given the following contents from a web search for the query <query>{query}</query>, generate a list of learnings from the contents.
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

    # LLMを使用して学習内容とフォローアップ質問を生成
    try:
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

        learnings = output_dict.get("learnings", [])
        followup_questions = output_dict.get("followup_questions", [])

        # 結果の検証
        if not learnings:
            print("  Warning: No learnings generated")
            learnings = ["No clear learnings could be extracted from the search results."]

        if not followup_questions:
            print("  Warning: No follow-up questions generated")
            followup_questions = [
                f"What are the key aspects of {query}?",
                f"What are the latest developments in {query}?",
                f"What are the practical applications of {query}?"
            ]

        return {
            "learnings": learnings,
            "followup_questions": followup_questions
        }

    except Exception as e:
        print(f"  Error generating learnings and follow-up questions: {e}")
        # エラーが発生した場合はデフォルト値を返す
        return {
            "learnings": ["Error processing search results."],
            "followup_questions": [
                f"What are the basics of {query}?",
                f"Why is {query} important?",
                f"What are the latest developments in {query}?"
            ]
        }


async def main():
    """テスト用のメイン関数"""
    llm_name = "gpt-4o-mini-2024-07-18"

    # テスト用のサンプルデータ
    sample_results = [
        SearchResponseItem(
            url="https://example.com/article1",
            markdown="This is some example content about artificial intelligence. AI has been advancing rapidly in recent years.",
            title="AI Advancements"
        ),
        SearchResponseItem(
            url="https://example.com/article2",
            markdown="Machine learning is a subset of AI that focuses on training models to learn from data.",
            title="Machine Learning Basics"
        )
    ]

    query = "artificial intelligence trends"
    print(f"Testing process_serp_result with query: '{query}'")

    try:
        processed_result = await process_serp_result(llm_name, query, sample_results)

        print("\nLearnings:")
        for i, learning in enumerate(processed_result.learnings, 1):
            print(f"{i}. {learning}")

        print("\nFollow-up questions:")
        for i, question in enumerate(processed_result.followup_questions, 1):
            print(f"{i}. {question}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
