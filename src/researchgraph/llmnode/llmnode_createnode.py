# %%
from llmlinks.llm_client import LLMClient
from llmlinks.compiler import LLMCompiler


if __name__ == "__main__":
    llm_name = "gpt-4o-2024-08-06"
    llm = LLMClient(llm_name)
    response = llm("こんにちわ")
    print(response)
    compiler = LLMCompiler(llm)
    source = """数学の問題を解くpythonプログラムを書く。

    問題についてじっくり考えたうえで、その思考過程と、対応するプログラムを書く。

    Args:
        problem (str): 解くべき問題。

    Returns:
        thought (str): 問題を解くために考えたこと。
        python (str): pythonプログラム。
    """

    compiled = compiler.compile(source)
    print(compiled)
