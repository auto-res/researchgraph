from typing import Literal
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from IPython.display import Image
import random


# Stateを宣言
class State(TypedDict):
    value: str


# Nodeを宣言
def node(state: State, config: RunnableConfig):
    # 更新するStateの値を返す
    return {"value2": "1"}


def node2(state: State, config: RunnableConfig):
    state["value2"] = str(int(state["value2"]) + 100)
    return state


def node3(state: State, config: RunnableConfig):
    state["value2"] = str(int(state["value2"]) + 500)
    return state


def routing(state: State, config: RunnableConfig) -> Literal["node2", "node3"]:
    # random_numが0なら次のpathは"node2"になり、1なら"node3"になる。
    random_num = random.randint(0, 1)
    if random_num == 0:
        return "node2"
    else:
        return "node3"


# Graphの作成
graph_builder = StateGraph(State)


# Nodeの追加
graph_builder.add_node("node", node)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)

# Nodeをedgeに追加
# graph_builder.add_edge("node", "node2")

graph_builder.add_conditional_edges(
    "node",
    routing,
)

# Graphの始点を宣言
graph_builder.set_entry_point("node")

# Graphの終点を宣言
graph_builder.set_finish_point("node2")

# Graphをコンパイル
graph = graph_builder.compile()

# Graphの実行(引数にはStateの初期値を渡す)

memory = {
    "value": "",
    "value2": "",
}

graph.invoke(memory, debug=True)


graph = graph_builder.compile()
image = Image(graph.get_graph().draw_mermaid_png())

with open("../data/research_architecture.png", "wb") as f:
    f.write(image.data)
