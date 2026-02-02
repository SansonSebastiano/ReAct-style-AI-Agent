from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State of the data analysis agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_code: str | None
    execution_output: str | None
    plot_path: str | None
    iteration_count: int