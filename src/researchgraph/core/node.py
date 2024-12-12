import logging
from abc import ABC, abstractmethod
from typing import Dict, Any


class NodeExecutionError(Exception):
    def __init__(self, node_name: str, message: str, original_error: Exception = None):
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node {node_name} execution failed: {message}")


class Node(ABC):
    """Base class of all nodes"""

    def __init__(
        self,
        input_key: list[str],
        output_key: list[str],
    ):
        """
        Node initialization
        """
        self.input_key = input_key
        self.output_key = output_key
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Initialized node with input: {self.input_key}, output: {self.output_key}")

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        - Abstract methods that define the specific processing for each node
        - Must be implemented in a subclass
        """
        pass

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node execution
        """
        try:
            self.logger.debug("Starting node execution")
            self.before_execute()

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
        """
        - Processing performed before execution
        - Can be overridden in a derived class
        """
        self.logger.debug("Running pre-execution hooks")

    def after_execute(self):
        """
        - Process to be executed after execution
        - Can be overridden in a derived class
        """
        self.logger.debug("Running post-execution hooks")
