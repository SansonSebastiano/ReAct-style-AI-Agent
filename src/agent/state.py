from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """State for the ReAct agent."""
    messages: Annotated[list[BaseMessage], operator.add]    # To accumulate conversation history between user and agent
    iteration: int          # To track the iterations in the ReAct loop
    max_iterations: int     # Maximum allowed iterations
    task_complete: bool     # Whether the task is complete
