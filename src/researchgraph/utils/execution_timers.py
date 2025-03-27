import time
from functools import wraps
from typing import TypedDict

class ExecutionTimeState(TypedDict, total=False):
    execution_time: dict[str, dict[str, list[float]]]


def time_node(subgraph_name: str, node_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, state, *args, **kwargs):
            header = f"[{subgraph_name}.{node_name}]".ljust(40)
            print(f"{header} 開始")
            start = time.time()
            result = func(self, state, *args, **kwargs)
            end = time.time()
            duration = round(end - start, 4)

            timings = state.get("execution_time", {})
            subgraph_log = timings.get(subgraph_name, {})
            durations = subgraph_log.get(node_name, [])
            durations.append(duration)
            subgraph_log[node_name] = durations
            timings[subgraph_name] = subgraph_log
            state["execution_time"] = timings

            print(f"{header} 実行時間: {duration:7.4f} 秒")
            return result
        return wrapper
    return decorator


def time_subgraph(subgraph_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            header = f"[{subgraph_name}]".ljust(40)
            print(f"{header} 開始")
            start = time.time()
            result = func(state, *args, **kwargs)
            end = time.time()
            duration = round(end - start, 4)

            timings = state.get("execution_time", {})
            subgraph_log = timings.get(subgraph_name, {})
            durations = subgraph_log.get("__subgraph_total__", [])
            durations.append(duration)
            subgraph_log["__subgraph_total__"] = durations
            timings[subgraph_name] = subgraph_log
            state["execution_time"] = timings

            print(f"{header} 実行時間: {duration:7.4f} 秒")
            return result
        return wrapper
    return decorator