import logging
import sys
import os
from dotenv import load_dotenv
from pathlib import Path

from src.agent.react_agent import DataAnalysisAgent

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
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n❌ Error: Google API key not found!")
        print("\nPlease set one of the following environment variables:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print("  or")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://aistudio.google.com/app/apikey\n")
        return False
    
    return True

def main():
    """Main entry point for the ReAct agent application."""
    
    print("=" * 80)
    print("ReAct Data Analysis Agent")
    print("=" * 80)
    print("\nThis agent can help you with data analysis and Plotly visualizations.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    if not check_api_key():
        return

    # Initialize the agent
    try:
        agent = DataAnalysisAgent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        print(f"Error: Could not initialize agent. Check your API credentials.")
        return
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            print("\n" + "-" * 80)
            user_query = input("\nYour request: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_query:
                print("Please enter a request.")
                continue
            
            # Run the agent
            print("\n🤖 Agent is working...\n")
            logger.info(f"Processing user query: {user_query}")
            
            final_state = agent.run(user_query)
            
            # Display results
            print("\n" + "=" * 80)
            print("AGENT RESPONSE")
            print("=" * 80)
            
            # Print the conversation history
            for i, message in enumerate(final_state["messages"]):
                role = message.__class__.__name__.replace("Message", "")
                content = message.content
                
                # Skip internal CODE_TO_EXECUTE messages
                if "CODE_TO_EXECUTE:" in content:
                    continue
                
                print(f"\n[{role}]")
                print(content)
                print("-" * 40)
            
            # Check for generated plots
            output_dir = Path("output")
            if output_dir.exists():
                plots = sorted(output_dir.glob("*.html"))
                if plots:
                    latest_plot = plots[-1]
                    print(f"\n✅ Plot generated: {latest_plot}")
                    print(f"   Open it in your browser to view the visualization.\n")
            
            # Show iteration count
            iterations_used = final_state.get("iteration", 0)
            max_iterations = final_state.get("max_iterations", 10)
            task_complete = final_state.get("task_complete", False)
            
            status = "✅ Completed" if task_complete else "⚠️ Stopped (max iterations)"
            print(f"\nStatus: {status}")
            print(f"Iterations used: {iterations_used}/{max_iterations}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def run_example():
    """Run a predefined example for testing."""
    print("Running example query...\n")
    
    # Check API key before running example
    if not check_api_key():
        return
    
    try:
        agent = DataAnalysisAgent()
        
        example_query = """
        Create a line chart showing the trend of sales data over 12 months.
        Use sample data: January through December with values [100, 120, 140, 130, 150, 170, 180, 190, 200, 210, 220, 240].
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
        print(f"\n❌ Example failed: {e}\n")


if __name__ == "__main__":
    # Check if running in example mode
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        run_example()
    else:
        main()