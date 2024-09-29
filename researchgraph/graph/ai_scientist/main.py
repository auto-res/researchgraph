import argparse
import os
import os.path as osp

from ai_scientist.generate_ideas import IdeaGenerationComponent
from ai_scientist.execute_idea import IdeaExecutionComponent

NUM_REFLECTIONS = 3


class AIscientist:
    def __call__(self, *args: argparse.Any, **kwds: argparse.Any) -> argparse.Any:
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20240620",
        choices=[
            "claude-3-5-sonnet-20240620",
            "gpt-4o-2024-05-13",
            "deepseek-coder-v2-0724",
            "llama3.1-405b",
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/anthropic.claude-3-opus-20240229-v1:0",
            "vertex_ai/claude-3-opus@20240229",
            "vertex_ai/claude-3-5-sonnet@20240620",
            "vertex_ai/claude-3-sonnet@20240229",
            "vertex_ai/claude-3-haiku@20240307",
        ],
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="latex",
        choices=["latex"],
        help="What format to use for writeup",
    )
    parser.add_argument(
        "--improvement", action="store_true", help="Improve based on reviews."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas", type=int, default=50, help="Number of ideas to generate"
    )
    return parser.parse_args()


def get_client(model):
    if model.startswith("claude"):
        import anthropic

        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        import anthropic

        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        return anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        ), client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        import anthropic

        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif model in ["gpt-4o-2024-05-13", "deepseek-coder-v2-0724", "llama3.1-405b"]:
        import openai

        print(f"Using OpenAI API with model {model}.")
        if model == "deepseek-coder-v2-0724":
            return openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            ), model
        elif model == "llama3.1-405b":
            return openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ), "meta-llama/llama-3.1-405b-instruct"
        else:
            return openai.OpenAI(), model
    else:
        raise ValueError(f"Model {model} not supported.")


def main(args):
    client, client_model = get_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    idea_generator = IdeaGenerationComponent()
    memory = {}
    memory = idea_generator(
        base_dir,
        client,
        args.model,
        memory,
        args.skip_idea_generation,
        args.num_ideas,
        NUM_REFLECTIONS,
    )
    ideas = memory["ideas"]

    idea_executor = IdeaExecutionComponent()
    for idea in ideas:
        print(f"Processing idea: {idea['Name']}")
        try:
            memory = idea_executor(
                base_dir,
                results_dir,
                idea,
                args.model,
                client,
                client_model,
                args.writeup,
                args.improvement,
                memory,
            )
            print(
                f"Completed idea: {idea['Name']}, Success: {memory['is_idea_execution_successful']}"
            )
        except Exception as e:
            print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")

    print("All ideas evaluated.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
