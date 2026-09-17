"""
Base Agent Interface for PAAF Multi-Agent Framework.
Handles agent execution, prompt construction, and LLM reasoning simulation.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, name: str, role_description: str):
        self.name = name
        self.role_description = role_description

    @abstractmethod
    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data within the orchestrator context and return structured output."""
        pass

    def format_log(self, message: str) -> str:
        return f"[{self.name}]: {message}"
