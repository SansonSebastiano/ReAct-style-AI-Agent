from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    State for the ReAct agent.

    Attributes:
        messages (list[BaseMessage]): Conversation history between the user and the agent.
        iteration (int): Current iteration count of the agent's reasoning loop.
        max_iterations (int): Maximum allowed iterations for the agent.
        task_complete (bool): Flag indicating whether the task has been completed.
    """
    messages: Annotated[list[BaseMessage], operator.add]   
    iteration: int          
    max_iterations: int     
    task_complete: bool     
