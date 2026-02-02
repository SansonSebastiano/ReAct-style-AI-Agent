from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .sandbox import CodeSandbox

sandbox = CodeSandbox(timeout=30)

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

    result = sandbox.execute_code(code)

    if result["success"]:
        output = f"Code executed successfully.\n{result['stdout']}"
        if result["plot_path"]:
            output += f"\nGenerated plot available at: {result['plot_path']}"
        return output
    else:
        return f"Code execution failed with error:\n{result['stderr']}"