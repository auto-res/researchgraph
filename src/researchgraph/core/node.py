from abc import ABC, abstractmethod


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
        print(f"input: {self.input_key}")
        print(f"output: {self.output_key}")

    @abstractmethod
    def execute(self, state) -> dict:
        """
        - Abstract methods that define the specific processing for each node
        - Must be implemented in a subclass
        """
        pass

    def __call__(self, state):
        """
        Node execution
        """
        self.before_execute()
        result = self.execute(state)
        self.after_execute()
        return result

    def before_execute(self):
        """
        - Processing performed before execution
        - Can be overridden in a derived class
        """

    def after_execute(self):
        """
        - Process to be executed after execution
        - Can be overridden in a derived class
        """
