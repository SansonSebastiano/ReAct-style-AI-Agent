import subprocess
import tempfile
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CodeSandbox:
    """A sandbox environment to safely execute generated code."""

    def __init__(self, timeout: int = 30, output_dir: str = "output"):
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def execute_code(self, code: str):
        """Execute the given code in a sandboxed environment.

        Args:
            code (str): The code to execute.

        Returns:
            dict: A dictionary containing execution output and any generated files. The keys of the dictionary are: `success` (bool), `stdout` (str), `stderr` (str), `plot_path` (str | None).
        """    

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                code_file = Path(temp_dir) / "script.py"
                code_file.write_text(code)

                result = subprocess.run(
                    ["python", str(code_file)],
                    cwd=temp_dir,
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
                )

                plot_path = None    
                output_file = Path(temp_dir) / "output.html"
                if output_file.exists():
                    import time
                    plot_name = f"plot_{int(time.time())}.html"
                    plot_path = self.output_dir / plot_name
                    output_file.rename(plot_path)

                logger.info(f"Code executed successfully. Output path: {plot_path}")

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "plot_path": str(plot_path) if plot_path else None
                }
            
            except subprocess.TimeoutExpired:
                logger.error(f"Code execution timed out. {self.timeout}s elapsed.")

                return {
                    "success": False,   
                    "stdout": "",
                    "stderr": "Execution timeout expired ({self.timeout}s).",
                    "plot_path": None
                }
            except Exception as e:
                logger.error(f"Error during code execution: {e}")

                return {
                    "success": False,
                    "stdout": "",
                    "stderr": str(e),
                    "plot_path": None
                }