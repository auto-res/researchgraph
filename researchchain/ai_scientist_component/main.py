import openai
import os.path as osp
import argparse
import torch
import os

from ai_scientist.generate_ideas import IdeaGenerationComponent
from ai_scientist.execute_idea import IdeaExecutionComponent

NUM_REFLECTIONS = 3


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
    # add type of experiment (nanoGPT, Boston, etc.)
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
            # Anthropic Claude models via Amazon Bedrock
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/anthropic.claude-3-opus-20240229-v1:0"
            # Anthropic Claude models Vertex AI
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
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=50,
        help="Number of ideas to generate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Create client
    if args.model == "claude-3-5-sonnet-20240620":
        import anthropic

        print(f"Using Anthropic API with model {args.model}.")
        client_model = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic()
    elif args.model.startswith("bedrock") and "claude" in args.model:
        import anthropic

        # Expects: bedrock/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
    elif args.model.startswith("vertex_ai") and "claude" in args.model:
        import anthropic

        # Expects: vertex_ai/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Vertex AI with model {client_model}.")
        client = anthropic.AnthropicVertex()
    elif args.model == "gpt-4o-2024-05-13":
        import openai

        print(f"Using OpenAI API with model {args.model}.")
        client_model = "gpt-4o-2024-05-13"
        client = openai.OpenAI()
    elif args.model == "deepseek-coder-v2-0724":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "deepseek-coder-v2-0724"
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    elif args.model == "llama3.1-405b":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "meta-llama/llama-3.1-405b-instruct"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Model {args.model} not supported.")

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    # NOTE: IdeaGenerationComponent ベースに変更
    memory_ = {}
    idea_generator = IdeaGenerationComponent()
    memory = idea_generator(
        base_dir=base_dir,
        client=client,
        model=args.model,
        memory_=memory_,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
    )
    ideas = memory["ideas"]

    for idea in ideas:
        print(f"Processing idea: {idea['Name']}")
        try:
            idea_executor = IdeaExecutionComponent()
            memory_ = idea_executor(
                base_dir,
                results_dir,
                idea,
                args.model,
                client,
                client_model,
                args.writeup,
                args.improvement,
                memory_,
            )

            print(f"Completed idea: {idea['Name']}, Success: {memory_["is_idea_execution_successful"]}")
        except Exception as e:
            print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")

    print("All ideas evaluated.")