import os
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
import nest_asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
# NOTE: Adjust this import to use a different embedding model
from models import OPENAI_EMBEDDING_MODEL as EMBEDDING_MODEL

LANGGRAPH_DOCS = [
    "https://langchain-ai.github.io/langgraph/",
    "https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/",
    "https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting/",
    "https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/",
    "https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/",
    "https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/",
    "https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/",
    "https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/",
    "https://langchain-ai.github.io/langgraph/tutorials/rewoo/rewoo/",
    "https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/LLMCompiler/",
    "https://langchain-ai.github.io/langgraph/concepts/high_level/",
    "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/",
    "https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/",
    "https://langchain-ai.github.io/langgraph/concepts/multi_agent/",
    "https://langchain-ai.github.io/langgraph/concepts/persistence/",
    "https://langchain-ai.github.io/langgraph/concepts/streaming/",
    "https://langchain-ai.github.io/langgraph/concepts/faq/"
]

def get_langgraph_docs_retriever():
    # If there is a vectorstore at this path, early return as it is already persisted
    if os.path.exists("langgraph-docs-db"):
        print("Loading vectorstore from disk...")
        vectorstore = Chroma(
            collection_name="langgraph-docs",
            embedding_function=EMBEDDING_MODEL,
            persist_directory="langgraph-docs-db"
        )
        return vectorstore.as_retriever(lambda_mult=0)

    # Otherwise, load the documents and persist to the vectorstore
    docs = [WebBaseLoader(url).load() for url in LANGGRAPH_DOCS]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    vectorstore = Chroma(
        collection_name="langgraph-docs",
        embedding_function=EMBEDDING_MODEL,
        persist_directory="langgraph-docs-db"
    )
    vectorstore.add_documents(doc_splits)
    print("Vectorstore created and persisted to disk")
    return vectorstore.as_retriever(lambda_mult=0)

def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with fallback rendering.
    
    Handles timeout errors from mermaid.ink by falling back to pyppeteer.
    
    Args:
        graph: The LangGraph object that has a get_graph() method
    """
    from IPython.display import Image
    try:
        # Try the default renderer first
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        # Fall back to pyppeteer if the default renderer fails
        import nest_asyncio
        nest_asyncio.apply()
        from langchain_core.runnables.graph import MermaidDrawMethod
        return Image(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))