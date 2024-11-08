from typing import Literal
from typing_extensions import TypedDict


class State(TypedDict):
    method_1_executable: str
    method_1_completion: str
    method_1_code_experiment: list
    new_method_executable: str
    new_method_code: list
    new_method_completion: str
    new_metho_code_experiment: list


def branchcontroller1(state: State) -> Literal["coder1", "__end__"]:
    # def branchcontroller1(state: State) -> Literal["coder1", "csvretriever1"]:
    if state["method_1_executable"] == "True":
        return "coder1"
    else:
        return "__end__"
        # return "csvretriever1"


def branchcontroller2(
    state: State,
) -> Literal["debugger1", "comparator", "csvretriever1"]:
    if state["method_1_completion"] == "True":
        return "comparator"
    else:
        if len(state["method_1_code_experiment"]) <= 4:
            return "debugger1"
        else:
            return "csvretriever1"


def branchcontroller3(state: State) -> Literal["coder2", "__end__"]:
    # def branchcontroller3(state: State) -> Literal["coder2", "creator", "csvretriever2"]:
    if state["new_method_executable"] == "True":
        return "coder2"
    else:
        return "__end__"
    #     return "csvretriever2"
    # if len(state["new_method_code"]) <= 2:
    #     return "creator"
    # else:
    #     return "csvretriever2"


def branchcontroller4(
    state: State,
) -> Literal["debugger2", "comparator", "csvretriever2"]:
    if state["new_method_completion"] == "True":
        return "comparator"
    else:
        if len(state["new_metho_code_experiment"]) <= 4:
            return "debugger2"
        else:
            return "csvretriever2"


def branchcontroller5(state: State) -> Literal["csvretriever1", "coder1"]:
    if state["method_1_code_error"] == "True":
        return "coder1"
    else:
        return "verifier1"
