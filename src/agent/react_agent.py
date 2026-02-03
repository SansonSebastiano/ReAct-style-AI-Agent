from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import os

from src.agent.state import AgentState
from src.tools.code_executor import execute_python_code

logger = logging.getLogger(__name__)

class DataAnalysisAgent:
    """ReAct-style agent for data analysis and visualization."""

    def __init__(self, model_name: str = "gemini-2.5-flash", max_iterations: int = 10):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0,
            google_api_key=api_key
        )
        self.max_iterations = max_iterations
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("observe", self._observe_node)
        
        # Set entry point
        workflow.set_entry_point("plan")
        
        # Add edges
        workflow.add_edge("plan", "act")
        workflow.add_edge("act", "observe")
        workflow.add_conditional_edges(
            "observe",
            self._should_continue,
            {
                "continue": "plan",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Planning phase: Reason about what to do next."""
        logger.info(f"Planning (iteration {state['iteration']})")
        
        system_prompt = """You are a data analysis and visualization expert using the ReAct framework.

Your task is to help users with data analysis and create Plotly visualizations.

**Available Tools:**
- execute_python: Execute Python code for data analysis and visualization. The code should save plots as 'output.html' in the current directory.

**ReAct Process:**
1. **Thought**: Reason about what needs to be done
2. **Action**: Decide to execute Python code
3. **Observation**: Review the execution result

**Important:**
- Always save Plotly figures as 'output.html' using: fig.write_html('output.html')
- Import required libraries (pandas, plotly, numpy, etc.)
- Break complex tasks into steps if needed
- If execution fails, analyze the error and adjust your approach

When you have completed the task successfully, respond with "TASK_COMPLETE" followed by a summary."""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        response = self.llm.invoke(messages)
        logger.info(f"Plan: {response.content[:200]}...")
        
        return {
            "messages": [AIMessage(content=f"**Thought (Iteration {state['iteration']}):** {response.content}")],
            "iteration": state["iteration"],
            "max_iterations": state["max_iterations"],
            "task_complete": "TASK_COMPLETE" in response.content.upper()
        }
    
    def _act_node(self, state: AgentState) -> AgentState:
        """Action phase: Generate and prepare to execute code."""
        logger.info("Acting - generating code")
        
        # Check if task is already complete
        if state["task_complete"]:
            return {
                "messages": [],
                "iteration": state["iteration"],
                "max_iterations": state["max_iterations"],
                "task_complete": True
            }
        
        code_prompt = """Based on your thought process, generate the Python code to execute.

**Requirements:**
- Include all necessary imports
- Save Plotly figures as 'output.html' using: fig.write_html('output.html')
- Include error handling where appropriate
- Add comments to explain key steps

Respond with ONLY the Python code, wrapped in ```python code blocks."""

        messages = state["messages"] + [HumanMessage(content=code_prompt)]
        response = self.llm.invoke(messages)
        
        # Extract code from response
        code = self._extract_code(response.content)
        
        if not code:
            logger.warning("No code generated")
            return {
                "messages": [AIMessage(content="**Action:** No code to execute. Need to rethink approach.")],
                "iteration": state["iteration"],
                "max_iterations": state["max_iterations"],
                "task_complete": False
            }
        
        logger.info(f"Generated code:\n{code[:200]}...")
        
        return {
            "messages": [
                AIMessage(content=f"**Action:** Executing Python code:\n```python\n{code}\n```"),
                HumanMessage(content=f"CODE_TO_EXECUTE:{code}")
            ],
            "iteration": state["iteration"],
            "max_iterations": state["max_iterations"],
            "task_complete": False
        }
    
    def _observe_node(self, state: AgentState) -> AgentState:
        """Observation phase: Execute code and observe results."""
        logger.info("Observing - executing code")
        
        # Check if task is already complete
        if state["task_complete"]:
            return {
                "messages": [],
                "iteration": state["iteration"] + 1,
                "max_iterations": state["max_iterations"],
                "task_complete": True
            }
        
        # Find the code to execute from the last message
        code = None
        for msg in reversed(state["messages"]):
            if hasattr(msg, 'content') and "CODE_TO_EXECUTE:" in msg.content:
                code = msg.content.split("CODE_TO_EXECUTE:", 1)[1].strip()
                break
        
        if not code:
            observation = "No code found to execute."
        else:
            # Execute the code using the tool
            observation = execute_python_code.invoke({"code": code})

            task_complete = (
                "Code executed successfully" in observation 
                and "Generated plot available at:" in observation
            )

            if task_complete:
                logger.info("Task automatically marked complete after successful execution")
        
        return {
            "messages": [AIMessage(content=f"**Observation:** {observation}")],
            "iteration": state["iteration"] + 1,
            "max_iterations": state["max_iterations"],
            "task_complete": task_complete
        }
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Decide whether to continue the ReAct loop or end."""
        
        # Check if task is complete
        if state["task_complete"]:
            logger.info("Task marked as complete")
            return "end"
        
        # Check max iterations
        if state["iteration"] >= state["max_iterations"]:
            logger.warning(f"Max iterations ({state['max_iterations']}) reached")
            return "end"
        
        # Check if last observation indicates success
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and "Generated plot available at:" in last_message.content:
            logger.info("Plot generated successfully")
            return "end"
        
        return "continue"
    
    def _extract_code(self, content: str) -> str:
        """Extract Python code from markdown code blocks."""
        if "```python" in content:
            parts = content.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        elif "```" in content:
            parts = content.split("```")
            if len(parts) > 2:
                code = parts[1].strip()
                return code
        return content.strip()
    
    def run(self, user_query: str) -> dict:
        """Run the ReAct agent on a user query.
        
        Args:
            user_query: The user's request for data analysis or visualization
            
        Returns:
            dict: Final state containing messages and execution history
        """
        logger.info(f"Starting ReAct agent with query: {user_query}")
        
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "task_complete": False
        }
        
        final_state = self.graph.invoke(initial_state)
        
        logger.info("ReAct agent completed")
        return final_state