from datetime import date
import logging
from typing import Optional
from datetime import datetime
from src.LLM_SQL_Support.database import connect_database
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool
from src.LLM_SQL_Support.model_loader import load_model
@tool
def fetch_user_order_information(order_id:str) -> list[dict]:
    """Fetch all orders for the user along with corresponding product and shipment information.

    Returns:
        A list of dictionaries where each dictionary contains the order details,
        associated product details, and shipment information for each order belonging to the user.
    """

    conn = connect_database()
    cursor = conn.cursor()

    query = """
    SELECT
        Date,Status,Fulfilment,Sales Channel ,ship-service-level,Style,SKU,Category,Size,ASIN,Courier Status,
        Qty,currency,Amount,ship-city,ship-state,ship-postal-code,ship-country,promotion-ids,B2B,fulfilled-by
    FROM
         'amazon_sale_report'
    WHERE
       `Order ID`= ?
    """
    cursor.execute(query, (order_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results
@tool
def fetch_order_status(order_id: str):
    """Fetch the status of a specific order."""
    conn = connect_database()
    cursor = conn.cursor()

    query = "SELECT Status FROM 'amazon_sale_report' WHERE `Order ID`= ?"
    cursor.execute(query, (order_id,))
    status = cursor.fetchone()

    cursor.close()
    conn.close()

    if status:
        return status[0]
    else:
        return f"No order found with ID {order_id}."

@tool
def search_orders(
    order_id: Optional[str] = None,
    limit: int = 20,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> list[dict]:
    """Search for orders based on product name and order date range."""
    conn = connect_database()
    cursor = conn.cursor()

    query = "SELECT * FROM  'amazon_sale_report' WHERE 1 = 1"
    params = []

    if order_id:
        query += " AND `Order ID` LIKE ?"
        params.append(f"%{order_id}%")

    if start_time:
        query += " AND Date >= ?"
        params.append(start_time)

    if end_time:
        query += " AND Date <= ?"
        params.append(end_time)

    query += " LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results

@tool
def update_order_status(
    order_id: str, new_status: str
) -> str:
    """Update the user's order to a new status."""

    conn = connect_database()
    cursor = conn.cursor()

    # Ensure the user owns the order
    cursor.execute(
        "SELECT * FROM  'amazon_sale_report' WHERE `Order ID`= ? ", (order_id,)
    )
    current_order = cursor.fetchone()
    if not current_order:
        cursor.close()
        conn.close()
        return f"Order with ID {order_id} not found."

    # Update order status
    cursor.execute(
        "UPDATE orders SET Status = ? WHERE `Order ID` = ?", (new_status, order_id)
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "Order status successfully updated."


@tool
def cancel_order(order_id: str) -> str:
    """Cancel the user's order and remove it from the database."""
    conn = connect_database()
    cursor = conn.cursor()

    # Ensure the user owns the order
    cursor.execute(
        "SELECT * FROM  'amazon_sale_report' WHERE `Order ID` = ?", (order_id,)
    )
    current_order = cursor.fetchone()
    if not current_order:
        cursor.close()
        conn.close()
        return f"Order with ID {order_id} not found."

    # Cancel the order
    cursor.execute( "UPDATE orders SET Status = 'Cancelled' WHERE `Order ID` = ?", (order_id))
 
    conn.commit()

    cursor.close()
    conn.close()
    return "Order successfully cancelled."

def llm():
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""You are a customer service assistant. If the customer query includes an order ID, respond to the query based on the available information. If the query does not contain an order ID, ask the customer politely to provide their order ID for further assistance.

    Customer query: {query}
    """
    )
    llm=load_model()
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    llm_tool = Tool(
        name='Language Model',
        func=llm_chain.run,
        description="Handles general customer service inquiries. If the order ID is not provided, it will request the customer to provide their order ID to assist further. If it cannot find relevant information or lacks enough detail, it will respond with 'I don't know.'"
    )
    return llm_tool,llm
llm=llm()[1]
llm_tool = llm()[0]
tools=[llm_tool,cancel_order,update_order_status,search_orders,fetch_order_status]
