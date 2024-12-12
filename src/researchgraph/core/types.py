"""Type definitions for research graph components."""
from typing import Dict, List, Optional, Any, Union
from typing_extensions import TypedDict, NotRequired


class NodeState(TypedDict):
    """State information for node execution."""
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    status: str
    error: Optional[str]
    metadata: NotRequired[Dict[str, Any]]


class NodeConfig(TypedDict):
    """Configuration for node initialization."""
    input_key: List[str]
    output_key: List[str]
    cache_dir: Optional[str]
    cache_enabled: bool
    log_dir: Optional[str]
    log_enabled: bool


class NodeResult(TypedDict):
    """Result of node execution."""
    state: NodeState
    success: bool
    execution_time: float
    cached: bool


class RetryConfig(TypedDict):
    """Configuration for retry mechanism."""
    max_retries: int
    base_delay: float
    max_delay: float
    exponential_base: float


NodeInput = Union[str, int, float, bool, Dict[str, Any], List[Any]]
NodeOutput = Union[str, int, float, bool, Dict[str, Any], List[Any]]
