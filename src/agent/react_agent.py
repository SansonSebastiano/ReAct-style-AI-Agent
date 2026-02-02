from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.state import AgentState
from src.tools.code_executor import execute_python_code

class DataAnalysisAgent:
    def __init__(self, model: ChatGoogleGenerativeAI, tools: list):
        self.model = model.bind_tools(tools)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("planning", self._planning_node)
        workflow.add_node("action", self._action_node)
        workflow.add_node("observation", self._observation_node)

        workflow.set_entry_point("planning")
        workflow.add_conditional_edges(
            "planning",
            self._should_continue,
            {"continue": "action", "end": END},
        )
        workflow.add_edge("action", "observation")
        workflow.add_edge("observation", "planning")

        return workflow.compile()
    
    def _planning_node(self, state: AgentState):
        """Generate the next action based on the current state."""

        system_prompt = SystemMessage(content="""You are a data analysis assistant. Your task is to analyze data and generate visualizations based on user requests. Generate Python code to complete your task. Always think step-by-step before taking an action.""")

        response = self.model.invoke([system_prompt] + state["messages"])
        return {"message": [response], "iteration_count": state.get("iteration_count", 0) + 1}
    
    def _action_node(self, state: AgentState):
        """Execute the generated code and return the output."""

        last_message = state["messages"][-1]

        # Check if it's an AI message with tool calls
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}
        
        tool_call = last_message.tool_calls[0]
        selected_tool = {tool.name: tool for tool in self.tools}[tool_call["name"]]
        tool_output = selected_tool.invoke(tool_call["args"])
        
        return {"messages": [ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])]}
        
    def _observation_node(self, state: AgentState):
        """Process the output from the action node and update the state."""
        
        execution_output = state.get("execution_output", "No output available.")
        observation_message = f"Observation:\n{execution_output}"

        return {"message": [("system", observation_message)]}
    
    def _should_continue(self, state: AgentState):
        """Decide whether to continue planning or end the workflow."""

        iteration_limit = 5
        if state["iteration_count"] >= iteration_limit:
            return "end"
        
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            return "end"
        
        return "continue"
