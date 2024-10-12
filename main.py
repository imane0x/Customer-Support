import argparse
from src.LLM_SQL_Support.agent import Agent
from src.model_finetuning.fine_tuning import fine_tune_model  # Import fine-tuning logic

def main(fine_tune):
    if fine_tune:
        print("Fine-tuning the model... Please wait.")
        fine_tune_model()  # Assuming you have a function for fine-tuning the model
        print("Fine-tuning completed!")

    # Proceed with running the model
    react_agent, react_agent_executor = Agent()
    query = "I wanna cancel my order"
    print(f"Executing query: {query}")
    react_agent_executor.invoke({"input": query})

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the model with or without fine-tuning.")
    parser.add_argument('--fine_tune', action='store_true', help="Fine-tune the model before running it.")
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed argument
    main(args.fine_tune)
