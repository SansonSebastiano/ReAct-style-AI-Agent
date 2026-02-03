import logging
import sys
import os
from dotenv import load_dotenv
from pathlib import Path

from src.agent.react_agent import DataAnalysisAgent

# Change this query to test different scenarios

# Default example query: "Create a line chart showing the trend of sales data over 12 months. Use sample data: January through December with values [100, 120, 140, 130, 150, 170, 180, 190, 200, 210, 220, 240]."

# Alternative possible queries:
# - "Analyze the iris dataset and create a scatter plot of sepal length vs sepal width, colored by species."
EXAMPLE_QUERY = """
Create a line chart showing the trend of sales data over 12 months. Use sample data: January through December with values [100, 120, 140, 130, 150, 170, 180, 190, 200, 210, 220, 240].
"""

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/agent.log')
    ]
)

logger = logging.getLogger(__name__)

def check_api_key():
    """Check if the required API key is set."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("\nError: Google API key not found!")
        print("\nPlease set one of the GOOGLE_API_KEY environment variable in the `.env` file:")
        print("\nGet your API key from: https://aistudio.google.com/app/apikey\n")
        return False
    
    return True

def main():    
    if not check_api_key():
        return
    
    run_example()


def run_example():
    """Run a predefined example for testing."""
    print("Running example query...\n")

    # Initialize the agent
    try:
        agent = DataAnalysisAgent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        print(f"Error: Could not initialize agent. Check your API credentials.")
        return

    # Run the agent with the example query
    try:
        example_query = f"""
        {EXAMPLE_QUERY.strip()}
        Save it as output.html.
        """
        print(f"Query: {example_query}\n")
        
        final_state = agent.run(example_query)
        
        print("\n" + "=" * 80)
        print("Example completed!")
        print(f"Iterations: {final_state['iteration']}")
        print(f"Task complete: {final_state['task_complete']}")
        
        output_dir = Path("output")
        if output_dir.exists():
            plots = list(output_dir.glob("*.html"))
            if plots:
                print(f"Generated plot: {plots[-1]}")
    
    except Exception as e:
        logger.error(f"Example execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()