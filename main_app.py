import streamlit as st
import transformers
from huggingface_hub import notebook_login
from langchain_huggingface import HuggingFacePipeline
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
import torch
from langchain_huggingface.llms import HuggingFacePipeline



# Set up the LLM
llm_custom = HuggingFacePipeline.from_model_id(
    model_id="im21/Customer_Support_Mistral",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 1000},
)

# Set page configuration for Streamlit
st.set_page_config(page_title="Customer Support Chat", page_icon="💬", layout="centered")
st.title("Chat with the Customer Support Assistant 💬")
st.info("Ask me about your orders and get assistance!", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! How can I assist you with your orders today?"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    db_path = "data/Amazon_Sale_Report.db"
    return db_path  # Return the database path for use in tools

db_path = load_data()

# Define tools for the assistant
@Tool
def fetch_order_status(order_id: str):
    """Fetch the status of a specific order."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT Status FROM amazon_sale_report WHERE "Order ID" = ?', (order_id,))
    status = cursor.fetchone()
    cursor.close()
    conn.close()
    return status[0] if status else f"No order found with ID {order_id}."

@Tool
def fetch_order_details(order_id: str):
    """Fetch detailed information of a specific order."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT product_name, quantity, price, shipping_address, delivery_date
        FROM amazon_sale_report
        WHERE "Order ID" = ?
    """, (order_id,))
    details = cursor.fetchone()
    cursor.close()
    conn.close()
    if details:
        return {
            "Product Name": details[0],
            "Quantity": details[1],
            "Price": details[2],
            "Shipping Address": details[3],
            "Delivery Date": details[4]
        }
    else:
        return f"No order details found for Order ID {order_id}."

# Other tools can be defined similarly...

# Set up the conversational agent
memory = ConversationBufferMemory(memory_key="chat_history")
tools = [fetch_order_status, fetch_order_details]  # Add other tools as needed

conversational_agent = initialize_agent(
    agent='conversational-react-description',
    tools=tools,
    llm=llm_custom,
    verbose=True,
    memory=memory,
)

# User input and interaction
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversational_agent(prompt)
            response_message = response["messages"][-1]["content"]
            st.write(response_message)
            st.session_state.messages.append({"role": "assistant", "content": response_message})
