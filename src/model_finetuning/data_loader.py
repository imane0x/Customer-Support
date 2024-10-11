from datasets import load_dataset
import config


def load_customer_support_dataset(tokenizer):
    """Loads the customer support dataset and formats it for model training."""
    customer_service_prompt = """
    Below is a customer query followed by additional context. 
    Please respond in a helpful and professional manner to address the customer's concern.

    ### Query:
    {}

    ### Response:
    {}"""
    
    def format_prompts(examples):
        inputs = examples["query"]
        outputs = examples["response"]
        texts = [
            customer_service_prompt.format(input, output) + tokenizer.eos_token
            for input, output in zip(inputs, outputs)
        ]
        return {"text": texts}

    dataset = load_dataset(config.DATASET_NAME, split="train")
    formatted_dataset = dataset.map(format_prompts, batched=True)
    
    return formatted_dataset
