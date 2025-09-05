from agents.utils import llm, get_engine_for_chinook_db
from langchain_community.utilities.sql_database import SQLDatabase
from typing_extensions import TypedDict
from typing import Annotated, NotRequired
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import InjectedState

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

from langchain_core.tools import tool

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class State(InputState):
    customer_id: NotRequired[int]
    loaded_memory: NotRequired[str]
    remaining_steps: NotRequired[RemainingSteps]

@tool 
def get_invoices_by_customer_sorted_by_date(customer_id: Annotated[int, InjectedState("customer_id")]) -> list[dict]:
    """
    Look up all invoices for a customer using their ID, the customer ID is in a state variable, so you will not see it in the message history.
    The invoices are sorted in descending order by invoice date, which helps when the customer wants to view their most recent/oldest invoice, or if 
    they want to view invoices within a specific date range.
    
    Returns:
        list[dict]: A list of invoices for the customer.
    """
    # customer_id = state.get("customer_id", "Unknown user")
    return db.run(f"SELECT * FROM Invoice WHERE CustomerId = {customer_id} ORDER BY InvoiceDate DESC;")


@tool 
def get_invoices_sorted_by_unit_price(customer_id: Annotated[int, InjectedState("customer_id")]) -> list[dict]:
    """
    Use this tool when the customer wants to know the details of one of their invoices based on the unit price/cost of the invoice.
    This tool looks up all invoices for a customer, and sorts the unit price from highest to lowest. In order to find the invoice associated with the customer, 
    we need to know the customer ID. The customer ID is in a state variable, so you will not see it in the message history.

    Returns:
        list[dict]: A list of invoices sorted by unit price.
    """
    # customer_id = state.get("customer_id", "Unknown user")
    query = f"""
        SELECT Invoice.*, InvoiceLine.UnitPrice
        FROM Invoice
        JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        WHERE Invoice.CustomerId = {customer_id}
        ORDER BY InvoiceLine.UnitPrice DESC;
    """
    return db.run(query)


@tool
def get_employee_by_invoice_and_customer(invoice_id: int, customer_id: Annotated[int, InjectedState("customer_id")]) -> dict:
    """
    This tool will take in an invoice ID and a customer ID and return the employee information associated with the invoice.
    The customer ID is in a state variable, so you will not see it in the message history.
    Args:
        invoice_id (int): The ID of the specific invoice.

    Returns:
        dict: Information about the employee associated with the invoice.
    """
    # customer_id = state.get("customer_id", "Unknown user")
    query = f"""
        SELECT Employee.FirstName, Employee.Title, Employee.Email
        FROM Employee
        JOIN Customer ON Customer.SupportRepId = Employee.EmployeeId
        JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
        WHERE Invoice.InvoiceId = ({invoice_id}) AND Invoice.CustomerId = ({customer_id});
    """
    
    employee_info = db.run(query, include_columns=True)
    
    if not employee_info:
        return f"No employee found for invoice ID {invoice_id} and customer identifier {customer_id}."
    return employee_info

invoice_tools = [get_invoices_by_customer_sorted_by_date, get_invoices_sorted_by_unit_price, get_employee_by_invoice_and_customer]

invoice_subagent_prompt = """
    You are a subagent among a team of assistants. You are specialized for retrieving and processing invoice information. You are routed for invoice-related portion of the questions, so only respond to them.. 

    You have access to three tools. These tools enable you to retrieve and process invoice information from the database. Here are the tools:
    - get_invoices_by_customer_sorted_by_date: This tool retrieves all invoices for a customer, sorted by invoice date.
    - get_invoices_sorted_by_unit_price: This tool retrieves all invoices for a customer, sorted by unit price.
    - get_employee_by_invoice_and_customer: This tool retrieves the employee information associated with an invoice and a customer.
    
    If you are unable to retrieve the invoice information, inform the customer you are unable to retrieve the information, and ask if they would like to search for something else.
    
    CORE RESPONSIBILITIES:
    - Retrieve and process invoice information from the database
    - Provide detailed information about invoices, including customer details, invoice dates, total amounts, employees associated with the invoice, etc. when the customer asks for it.
    - Always maintain a professional, friendly, and patient demeanor
    
    You may have additional context that you should use to help answer the customer's query. It will be provided to you below:
    """

from langgraph.prebuilt import create_react_agent

# Define the subagent 
graph = create_react_agent(llm, tools=invoice_tools, name="invoice_information_subagent",prompt=invoice_subagent_prompt, state_schema=State)
