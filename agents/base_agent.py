from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Research-tier Abstract Base Class for AetherScribe agents.
    Enforces a strict interface for identity and action matrices.
    """
    def __init__(self, name):
        self.name = name

    @property
    @abstractmethod
    def actions(self):
        """Returns the list of possible actions for the agent."""
        pass
