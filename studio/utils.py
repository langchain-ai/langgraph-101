from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langgraph.store.memory import InMemoryStore, PutOp
from langchain.embeddings import init_embeddings
from models import OPENAI_EMBEDDING_MODEL
# from models import AZURE_OPENAI_EMBEDDING_MODEL

RAG_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:"""

RAG_PROMPT_WITH_CHAT_HISTORY = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
The pre-existing conversation may provide important context to the question.
Use three sentences maximum and keep the answer concise.

Conversation: {conversation}
Context: {context} 
Answer:"""

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

def get_vector_store(id: str, namespace: str):
    embeddings = init_embeddings("openai:text-embedding-3-small")
    store = InMemoryStore(index={"embed": embeddings, "dims": 1536})

    namespace_for_memory = (id, namespace)
    existing_data = store.search(namespace_for_memory, limit=1)
    # If the data has already been added - then return the existing store
    if len(existing_data) > 0:    
        return store
    
    # If not, then add the documents!
    docs = [WebBaseLoader(url).load() for url in LANGGRAPH_DOCS]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(doc_splits)
    ops = [
        PutOp(
            namespace=(id, namespace),
            key=f"doc-{i}",
            value=split,
            index=["metadata.title", "metadata.description", "page_content"]
        )
        for i, split in enumerate(doc_splits)
    ]
    store.batch(ops)
    return store