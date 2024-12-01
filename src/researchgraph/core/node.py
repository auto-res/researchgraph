from abc import ABC, abstractmethod


class Node(ABC):
    """Base class of all nodes"""

    def __init__(
        self,
        input_variable: list[str],
        output_variable: list[str],
    ):
        """
        Node initialization
        """
        self.input_variable = input_variable
        self.output_variable = output_variable
        print(f"input: {self.input_variable}")
        print(f"output: {self.output_variable}")

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
