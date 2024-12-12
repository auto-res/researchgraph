import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from .cache import NodeCache
from .logging import NodeLogger
from .types import NodeState, NodeConfig, NodeResult, NodeInput, NodeOutput


class NodeExecutionError(Exception):
    def __init__(self, node_name: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(f"[{node_name}] {message}")
        self.original_error = original_error


class Node(ABC):
    """Base class of all nodes with caching and logging support."""

    def __init__(
        self,
        input_key: List[str],
        output_key: List[str],
        cache_dir: Optional[str] = None,
        cache_enabled: bool = True,
        log_dir: Optional[str] = None,
        log_enabled: bool = True,
    ) -> None:
        """Initialize node with optional caching and logging.

        Args:
            input_key: List of input keys required by this node
            output_key: List of output keys produced by this node
            cache_dir: Directory for caching results, if None caching is disabled
            cache_enabled: Whether to enable caching
            log_dir: Directory for logging, if None logging is disabled
            log_enabled: Whether to enable logging
        """
        self.input_key = input_key
        self.output_key = output_key
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Initialized node with input: {self.input_key}, output: {self.output_key}")

        # Initialize cache if enabled
        self.cache = None
        if cache_dir and cache_enabled:
            self.cache = NodeCache(cache_dir)
            self.logger.debug(f"Initialized cache in {cache_dir}")

        # Initialize logger if enabled
        self.node_logger = None
        if log_dir and log_enabled:
            self.node_logger = NodeLogger(log_dir)
            self.logger.debug(f"Initialized logger in {log_dir}")

    def _generate_cache_key(self, state: Dict[str, Any]) -> str:
        """Generate cache key from input state.

        Args:
            state: Current state dictionary

        Returns:
            Cache key string
        """
        # Sort input keys to ensure consistent cache keys
        key_parts = [f"{k}:{state[k]}" for k in sorted(self.input_key)]
        return f"{self.__class__.__name__}:{':'.join(key_parts)}"

    @abstractmethod
    def execute(self, state: Dict[str, NodeInput]) -> Dict[str, NodeOutput]:
        """Execute node operation.

        Args:
            state: Current state dictionary

        Returns:
            Updated state dictionary

        Raises:
            NodeExecutionError: When execution fails
        """
        pass

    def __call__(self, state: Dict[str, NodeInput]) -> Dict[str, NodeOutput]:
        """Execute node with caching and logging support.

        Args:
            state: Current state dictionary

        Returns:
            Updated state dictionary

        Raises:
            NodeExecutionError: When execution fails
        """
        start_time = time.time()
        node_name = self.__class__.__name__

        try:
            self.logger.debug("Starting node execution")
            if self.node_logger:
                self.node_logger.log_start(node_name, state)

            self.before_execute()

            # Check cache if enabled
            if self.cache:
                cache_key = self._generate_cache_key(state)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.debug("Returning cached result")
                    execution_time = time.time() - start_time
                    if self.node_logger:
                        self.node_logger.log_complete(
                            node_name,
                            cached_result,
                            execution_time,
                            {"cached": True}
                        )
                    return cached_result

            # Validate input keys
            missing_keys = [key for key in self.input_key if key not in state]
            if missing_keys:
                raise NodeExecutionError(
                    node_name,
                    f"Missing required input keys: {missing_keys}"
                )

            # Execute node
            result = self.execute(state)

            # Validate output keys
            missing_outputs = [key for key in self.output_key if key not in result]
            if missing_outputs:
                raise NodeExecutionError(
                    node_name,
                    f"Missing required output keys: {missing_outputs}"
                )

            # Cache result if enabled
            if self.cache:
                cache_key = self._generate_cache_key(state)
                self.cache.set(cache_key, result)

            execution_time = time.time() - start_time
            if self.node_logger:
                self.node_logger.log_complete(
                    node_name,
                    result,
                    execution_time,
                    {"cached": False}
                )

            self.after_execute()
            self.logger.debug("Node execution completed successfully")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            if self.node_logger:
                self.node_logger.log_error(node_name, e, state)

            if isinstance(e, NodeExecutionError):
                raise
            self.logger.error(f"Node execution failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name,
                str(e),
                original_error=e
            )

    def before_execute(self) -> None:
        """Process to be executed before main execution.

        Can be overridden in derived classes.
        """
        self.logger.debug("Running pre-execution hooks")

    def after_execute(self) -> None:
        """Process to be executed after main execution.

        Can be overridden in derived classes.
        """
        self.logger.debug("Running post-execution hooks")
