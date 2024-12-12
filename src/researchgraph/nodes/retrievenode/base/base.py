"""Base module for retrieve nodes with retry mechanism."""
import time
import logging
from typing import Any, Callable, Dict, TypeVar

from researchgraph.core.node import Node, NodeExecutionError

T = TypeVar('T')

class BaseRetrieveNode(Node):
    """Base class for all retrieve nodes with retry mechanism."""

    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
        max_retries: int = 3,
        retry_delay: int = 1,
    ):
        """Initialize BaseRetrieveNode.

        Args:
            input_key: List of input keys required by this node
            output_key: List of output keys produced by this node
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Initial delay between retries in seconds
        """
        super().__init__(input_key, output_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_with_retry(self, operation: Callable[[], T]) -> T:
        """Execute an operation with retry mechanism.

        Args:
            operation: Callable that performs the API operation

        Returns:
            Result of the operation

        Raises:
            NodeExecutionError: When all retry attempts fail
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    self.logger.error(
                        f"Operation failed after {self.max_retries} attempts",
                        exc_info=True
                    )
                    raise NodeExecutionError(
                        self.__class__.__name__,
                        f"Operation failed after {self.max_retries} attempts: {str(e)}",
                        original_error=e
                    )

                delay = self.retry_delay * (attempt + 1)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay} seconds: {str(e)}"
                )
                time.sleep(delay)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node operation with retry mechanism.

        This method should be implemented by subclasses to define their specific
        retrieval logic. The implementation should use execute_with_retry for
        any API calls.

        Args:
            state: Current state dictionary

        Returns:
            Updated state dictionary

        Raises:
            NodeExecutionError: When execution fails
        """
        raise NotImplementedError("Subclasses must implement execute method")
