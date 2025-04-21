from agents.utils import get_langgraph_docs_retriever, llm
from langchain.schema import Document
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import operator
from langchain_core.messages import AnyMessage, get_buffer_string, SystemMessage, HumanMessage


retriever = get_langgraph_docs_retriever()

class GraphState(TypedDict):
    question: str
    messages: Annotated[List[AnyMessage], operator.add]     # We now track a list of messages
    generation: str
    documents: List[Document]
    attempted_generations: int

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]     # We output messages now in our OutputState
    documents: List[Document]

from langchain_core.messages import HumanMessage

def retrieve_documents(state: GraphState):
    """
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE DOCUMENTS---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

RAG_PROMPT_WITH_CHAT_HISTORY = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
The pre-existing conversation may provide important context to the question.
Use three sentences maximum and keep the answer concise.

Existing Conversation:
{conversation}

Latest Question:
{question}

Additional Context from Documents:
{context} 

Answer:"""

def generate_response(state: GraphState):
    # We interrupt the graph, and ask the user for some additional context
    additional_context = interrupt("Do you have anything else to add that you think is relevant?")
    print("---GENERATE RESPONSE---")
    question = state["question"]
    documents = state["documents"]
    # For simplicity, we'll just append the additional context to the conversation history
    conversation = get_buffer_string(state["messages"]) + additional_context
    attempted_generations = state.get("attempted_generations", 0)
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    
    rag_prompt_formatted = RAG_PROMPT_WITH_CHAT_HISTORY.format(context=formatted_docs, conversation=conversation, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {
        "generation": generation,
        "attempted_generations": attempted_generations + 1
    }

class GradeDocuments(BaseModel):
    is_relevant: bool = Field(
        description="The document is relevant to the question, true or false"
    )

grade_documents_llm = llm.with_structured_output(GradeDocuments)
grade_documents_system_prompt = """You are a grader assessing relevance of a retrieved document to a conversation between a user and an AI assistant, and user's latest question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, definitely grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals that are not relevant at all. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_documents_prompt = "Here is the retrieved document: \n\n {document} \n\n Here is the conversation so far: \n\n {conversation} \n\n Here is the user question: \n\n {question}"
def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    conversation = get_buffer_string(state["messages"])

    filtered_docs = []
    for d in documents:
        grade_documents_prompt_formatted = grade_documents_prompt.format(document=d.page_content, question=question, conversation=conversation)
        score = grade_documents_llm.invoke(
            [SystemMessage(content=grade_documents_system_prompt)] + [HumanMessage(content=grade_documents_prompt_formatted)]
        )
        grade = score.is_relevant
        if grade:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs}

def decide_to_generate(state):
    """
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, END---"
        )
        return "none relevant"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "some relevant"
    
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    grounded_in_facts: bool = Field(
        description="Answer is grounded in the facts, true or false"
    )

grade_hallucinations_llm = llm.with_structured_output(GradeHallucinations)
grade_hallucinations_system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score true or false. True means that the answer is grounded in / supported by the set of facts."""
grade_hallucinations_prompt = "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"

ATTEMPTED_GENERATION_MAX = 3

def grade_hallucinations(state):
    print("---CHECK HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]
    attempted_generations = state["attempted_generations"]

    formatted_docs = "\n\n".join(doc.page_content for doc in documents)

    grade_hallucinations_prompt_formatted = grade_hallucinations_prompt.format(
        documents=formatted_docs,
        generation=generation
    )

    score = grade_hallucinations_llm.invoke(
        [SystemMessage(content=grade_hallucinations_system_prompt)] + [HumanMessage(content=grade_hallucinations_prompt_formatted)]
    )
    grade = score.grounded_in_facts

    # Check hallucination
    if grade:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "supported"
    elif attempted_generations >= ATTEMPTED_GENERATION_MAX:    # New condition!
        print("---DECISION: TOO MANY ATTEMPTS, GIVE UP---")
        raise RuntimeError("Too many attempted generations with hallucinations, giving up.")
        # return "give up"    # Note: We could also do this to silently fail
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
def configure_memory(state):
    question = state["question"]
    generation = state["generation"]
    return {
        "messages": [HumanMessage(content=question), generation],   # Add generation to our messages_list
        "attempted_generations": 0,   # Reset this value to 0
        "documents": []    # Reset documents to empty
    }

graph_builder = StateGraph(GraphState, input=InputState, output=OutputState)
graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("generate_response", generate_response)
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("configure_memory", configure_memory)    # New node for configuring memory

graph_builder.add_edge(START, "retrieve_documents")
graph_builder.add_edge("retrieve_documents", "grade_documents")
graph_builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "some relevant": "generate_response",
        "none relevant": END
    })
graph_builder.add_conditional_edges(
    "generate_response",
    grade_hallucinations,
    {
        "supported": "configure_memory",
        "not supported": "generate_response"
    })
graph_builder.add_edge("configure_memory", END)

graph = graph_builder.compile()