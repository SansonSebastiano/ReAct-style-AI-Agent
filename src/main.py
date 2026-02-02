import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.react_agent import DataAnalysisAgent
from src.tools.code_executor import execute_python_code
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("logs/agent.log")
    ]
)

def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    tools = [execute_python_code]
    agent = DataAnalysisAgent(model=llm, tools=tools)

    user_query = "Load the iris dataset and create a scatter plot of sepal length vs width"
    result = agent.graph.invoke({
        "messages": [("user", user_query)],
        "iteration_count": 0
    })

    print("Final Output:")
    last_message = result["messages"][-1]
    last_message.pretty_print()
    print(f"Total Iterations: {result['iteration_count']}")

if __name__ == "__main__":
    main()