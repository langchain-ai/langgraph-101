from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from azure.identity import InteractiveBrowserCredential

credential = InteractiveBrowserCredential()

def get_token():
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return token.token

RAG_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:"""

RAG_PROMPT_WITH_MESSAGES = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
The pre-existing conversation may provide important context to the question.
Use three sentences maximum and keep the answer concise.

Conversation: {conversation}
Context: {context} 
Answer:"""

# SiteMap loader
LANGGRAPH_DOCS = [
    "https://langchain-ai.github.io/langgraph/",
    "https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/",
    "https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting/",
    "https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/",
]

def get_vector_db_retriever():
    # Set embeddings
    # embd = AzureOpenAIEmbeddings(
    #     openai_api_version="2024-03-01-preview",
    #     azure_endpoint="https://deployment.openai.azure.com/",
    #     model="text-embedding-3-large",
    #     azure_ad_token_provider=get_token
    # )
    embd = OpenAIEmbeddings()
    # Docs to index
    urls = LANGGRAPH_DOCS
    # Load
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embd,
    )
    retriever = vectorstore.as_retriever(lambda_mult=0)
    return retriever