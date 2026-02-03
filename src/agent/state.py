from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """State for the ReAct agent."""
    messages: Annotated[list[BaseMessage], operator.add]
    iteration: int
    max_iterations: int
    task_complete: bool
