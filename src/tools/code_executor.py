from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging

from .sandbox import CodeSandbox

logger = logging.getLogger(__name__)
sandbox = CodeSandbox()     # Initialize the sandbox environment here to avoid overhead

# Input schema for the code execution tool
class CodeInput(BaseModel):
    code: str = Field(description="Python code to execute for data analysis and visualization.")

@tool("execute_python", args_schema=CodeInput)
def execute_python_code(code: str) -> str:
    """Execute the provided Python code in a sandboxed environment.

    Args:
        code (str): The Python code to execute.

    Returns:
        str: The output of the code execution or error message.    
    """

    logger.info(f"Executing code:\n{code[:200]}...")
    # Execute the code in the sandboxed environment
    result = sandbox.execute_code(code)

    # See README.md for explanation about this snippet
    # if result["success"]:
    #     logger.info("Code executed successfully.")
    #     if result["plot_path"]:
    #         logger.info(f"Generated plot available at: {result['plot_path']}")
    #         output = 1 if result["success"] and result["plot_path"] else 0
    # else:
    #     logger.error(f"Code execution failed with error:\n{result['stderr']}")
    #     return 0

    # Process the result and return appropriate message
    if result["success"]:   # Code executed successfully
        output = f"Code executed successfully.\n{result['stdout']}"
        if result["plot_path"]:     # Plot saved successfully
            output += f"\nGenerated plot available at: {result['plot_path']}"
        return output
    else:   # Execution failed
        return f"Code execution failed with error:\n{result['stderr']}"