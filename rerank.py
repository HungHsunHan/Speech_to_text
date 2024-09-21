import os

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Initialize Azure OpenAI language model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your specific deployment name
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


# Load and process the PDF file
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    return splits


# Example usage: Replace with your actual PDF file path
pdf_path = "/Users/hanhongxun/Desktop/2023.pdf"
document_chunks = process_pdf(pdf_path)

# Create a vector store
vectorstore = FAISS.from_documents(document_chunks, embeddings)

# Create a basic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Create a prompt for semantic reranking
prompt_template = """Given the following context and question, extract the relevant information:

Context: {context}

Question: {question}

Relevant Information:"""

llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

# Create a document compressor for semantic reranking
document_compressor = LLMChainExtractor(llm_chain=llm_chain)

# Create a contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=document_compressor, base_retriever=retriever
)


# Use the compression retriever in your RAG pipeline
def rag_with_reranking(query):
    relevant_docs = compression_retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    final_prompt = f"""Use the following context to answer the question. If you can't answer based on the context, say "I don't have enough information to answer that."

Context: {context}

Question: {query}

Answer:"""

    response = llm(final_prompt)
    return response


# Example usage
query = "What is the main topic of the PDF?"
answer = rag_with_reranking(query)
print(answer.content)
