import json
import os

from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI embeddings
# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment="text-embedding-3-small",
#     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# )

# Initialize Azure OpenAI language model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your specific deployment name
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_info",
            "description": "Retrieves customer information based on their customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The unique identifier for the customer.",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Retrieves the details of a specific order based on the order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The unique identifier for the order.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancels an order based on the provided order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The unique identifier for the order to be cancelled.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
]


def get_customer_info(customer_id):
    # Simulated customer data
    customers = {
        "C1": {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "123-456-7890",
        },
        "C2": {
            "name": "Jane Smith",
            "email": "jane@example.com",
            "phone": "987-654-3210",
        },
    }
    return customers.get(customer_id, "Customer not found")


def get_order_details(order_id):
    # Simulated order data
    orders = {
        "O1": {
            "id": "O1",
            "product": "Widget A",
            "quantity": 2,
            "price": 19.99,
            "status": "Shipped",
        },
        "O2": {
            "id": "O2",
            "product": "Gadget B",
            "quantity": 1,
            "price": 49.99,
            "status": "Processing",
        },
    }
    return orders.get(order_id, "Order not found")


def cancel_order(order_id):
    # Simulated order cancellation
    if order_id in ["O1", "O2"]:
        return f"Order {order_id} has been successfully cancelled."
    else:
        return f"Order {order_id} not found or cannot be cancelled."


def process_tool_call(tool_name, tool_input):
    if tool_name == "get_customer_info":
        return get_customer_info(tool_input["customer_id"])
    elif tool_name == "get_order_details":
        return get_order_details(tool_input["order_id"])
    elif tool_name == "cancel_order":
        return cancel_order(tool_input["order_id"])
    else:
        return f"Unknown tool: {tool_name}"


def chatbot_interaction(user_message):
    print(f"\n{'='*50}\nUser Message: {user_message}\n{'='*50}")

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = llm.invoke(messages, tools=tools)

        print(f"\nAI Response:")
        print(response.content)

        if not response.tool_calls:
            return response.content

        print(f"tool_calls: {response.tool_calls}")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            # tool_input = json.loads(tool_call["args"])
            tool_input = tool_call["args"]

            print(f"\nTool Used: {tool_name}")
            print(f"Tool Input: {tool_input}")

            tool_result = process_tool_call(tool_name, tool_input)

            print(f"\nTool Result: {tool_result}")

            messages.append(response)
            messages.append(
                {
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call["id"],
                }
            )


# Example usage
if __name__ == "__main__":
    user_input = "What's the order status for order O1?"
    chatbot_interaction(user_input)
