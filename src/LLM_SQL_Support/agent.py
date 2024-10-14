from src.LLM_SQL_Support.LangchainTools import tools
from langchain.agents import initialize_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
def Agent(llm):
   

    # Define the tools and their names
    tool = {
        "language_model": "Handles customer service inquiries.",
        "cancel_order": "Cancel the user's order and remove it from the database.",
        "fetch_order_status": "Fetch the status of a specific order.",
        "fetch_order_details": "Fetch detailed information of a specific order."
    }

    tool_names = list(tool.keys())

    # Create the prompt template
    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                """
                Answer the following questions as best you can. You have access to the following tools:\n
                {tools}\n
                Use the following format:\n
                Question: the input question you must answer\n
                Thought: you should always think about what to do\n
                Action: the action to take, should be one of {tool_names}\n
                Action Input: the input to the action\n
                Observation: the result of the action\n
                Thought: I now know the final answer\n
                Final Answer: the final answer to the original input question\n
                
                Begin!\n
                tools:{tools}
                Question: {input}\n
                Thought:{agent_scratchpad}
                """
            )
        ]

    )
    react_agent = create_react_agent(llm, tools=tools,prompt=assistant_prompt)
    # executes the logical steps we created
    react_agent_executor = AgentExecutor(
        agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

 

    return react_agent,react_agent_executor
