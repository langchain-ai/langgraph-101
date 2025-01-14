from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings
from utils import LANGGRAPH_DOCS
from typing_extensions import TypedDict, Optional
import asyncio
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    load_status: Optional[str]

async def populate_langgraph_documents(state, config, store):
    namespace_for_memory = ("1", "langgraph-docs")
    existing_data = await store.asearch(namespace_for_memory, limit=1)
    # If the data has already been added - then return the existing store
    if len(existing_data) > 0:    
        return {"load_status": "Already Loaded!"}
    
    # If not, then add the documents!
    docs = [WebBaseLoader(url).load() for url in LANGGRAPH_DOCS]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    tasks = []
    for i, split in enumerate(doc_splits):
        tasks.append(
            store.aput(
                namespace_for_memory,
                f"doc-{i}",
                {
                    "title": split.metadata["title"],
                    "description": split.metadata["description"],
                    "page_content": split.page_content
                }
            )
        )
    
    await asyncio.gather(*tasks)
    return {"load_status": "Loaded LangGraph Docs!"}

graph_builder = StateGraph(GraphState)
graph_builder.add_node("populate_langgraph_documents", populate_langgraph_documents)
graph_builder.add_edge(START, "populate_langgraph_documents")
graph_builder.add_edge("populate_langgraph_documents", END)
graph = graph_builder.compile()