import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker, ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
from pydub import AudioSegment

load_dotenv()

# Initialize Azure OpenAI embeddings
# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment="text-embedding-3-small",
#     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# )

# # Initialize Azure OpenAI language model
# llm = AzureChatOpenAI(
#     azure_deployment="gpt-3.5-turbo",
#     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# )
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
# os.environ["MISTRAL_API_KEY"] = "EUPVs9dpUGaKKYS85vZ8BrSmyrfBVvFj"
# llm = MistralAI(
#     model="open-mixtral-8x22b",
#     temperature=0.1,
#     api_key="T8KEefs80Y6hjA5VB7JHGlAp3yKFV2Gv",
# )


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)

agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, add_tool, subtract_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = AgentRunner(agent_worker)
response = agent.chat("What is (26 * 2) + 2024?")
print(response)
