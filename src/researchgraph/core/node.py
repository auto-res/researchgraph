import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from .cache import NodeCache


class NodeExecutionError(Exception):
    def __init__(self, node_name: str, message: str, original_error: Exception = None):
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node {node_name} execution failed: {message}")


class Node(ABC):
    """Base class of all nodes with caching support."""

    def __init__(
        self,
        input_key: List[str],
        output_key: List[str],
        cache_dir: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        """Initialize node with optional caching.

        Args:
            input_key: List of input keys required by this node
            output_key: List of output keys produced by this node
            cache_dir: Directory for caching results, if None caching is disabled
            cache_enabled: Whether to enable caching
        """
        self.input_key = input_key
        self.output_key = output_key
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Initialized node with input: {self.input_key}, output: {self.output_key}")

        self.cache = None
        if cache_dir and cache_enabled:
            self.cache = NodeCache(cache_dir)
            self.logger.debug(f"Initialized cache in {cache_dir}")

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
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node operation.

        Args:
            state: Current state dictionary

        Returns:
            Updated state dictionary

        Raises:
            NodeExecutionError: When execution fails
        """
        pass

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node with caching support.

        Args:
            state: Current state dictionary

        Returns:
            Updated state dictionary

        Raises:
            NodeExecutionError: When execution fails
        """
        try:
            self.logger.debug("Starting node execution")
            self.before_execute()

            # Check cache if enabled
            if self.cache:
                cache_key = self._generate_cache_key(state)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.debug("Returning cached result")
                    return cached_result

            # Validate input keys
            missing_keys = [key for key in self.input_key if key not in state]
            if missing_keys:
                raise NodeExecutionError(
                    self.__class__.__name__,
                    f"Missing required input keys: {missing_keys}"
                )

            result = self.execute(state)

            # Validate output keys
            missing_outputs = [key for key in self.output_key if key not in result]
            if missing_outputs:
                raise NodeExecutionError(
                    self.__class__.__name__,
                    f"Missing required output keys: {missing_outputs}"
                )

            # Cache result if enabled
            if self.cache:
                cache_key = self._generate_cache_key(state)
                self.cache.set(cache_key, result)

            self.after_execute()
            self.logger.debug("Node execution completed successfully")
            return result

        except Exception as e:
            if isinstance(e, NodeExecutionError):
                raise
            self.logger.error(f"Node execution failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                self.__class__.__name__,
                str(e),
                original_error=e
            )

    def before_execute(self):
        """Process to be executed before main execution.

        Can be overridden in derived classes.
        """
        self.logger.debug("Running pre-execution hooks")

    def after_execute(self):
        """Process to be executed after main execution.

        Can be overridden in derived classes.
        """
        self.logger.debug("Running post-execution hooks")
