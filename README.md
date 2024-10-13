# Customer Support LLM with SQL Database Integration

This project uses a fine-tuned [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model to manage customer support queries, integrated with an SQL database for order management. The system allows responding to queries, checking order statuses, and retrieving user order histories.

## Features

- **Fine-tuned LLM**: The model has been fine-tuned using a [customer support dataset](https://huggingface.co/datasets/Kaludi/Customer-Support-Responses) to ensure it handles customer interactions effectively.
- **Order Management**: SQL integration provides the capability to fetch and update order information from the database, enabling real-time order status tracking, shipment updates, and customer history retrieval. To test the model, an Amazon sales report DB was used to validate these functionalities.

### **Setup**

-  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### **Usage**

- **Run without fine-tuning**:
    ```bash
    python main.py
    ```
- **Run with fine-tuning**:
    ```bash
    python main.py --fine_tune
    ```
