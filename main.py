from src.LLM_SQL_Support.agent import Agent

def main():
    react_agent,react_agent_executor = Agent()
    query = "I wanna cancel my order"
    react_agent_executor.invoke({"input": query})


if __name__ == "__main__":
    main()
