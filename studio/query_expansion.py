from dotenv import load_dotenv
from utils import get_vector_db_retriever, RAG_PROMPT
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langgraph.constants import Send
from collections import Counter

ATTEMPTED_GENERATION_MAX = 3

load_dotenv(dotenv_path="./.env", override=True)

retriever = get_vector_db_retriever(id="3")

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

def custom_documents_reducer(existing, update):
    # If we passed in a dictionary that asks for "overwrite", then we return the updated documents only
    if isinstance(update, dict) and update["type"] == "overwrite":
        return update["documents"]

    # Otherwise, we simple add the lists
    return existing + update

class GraphState(TypedDict):
    question: str
    rewritten_queries: Optional[List[str]]
    generation: Optional[str]
    documents: Annotated[Optional[List[Document]], custom_documents_reducer]
    attempted_generations: Optional[int]

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    generation: str
    documents: List[Document]

from pydantic import BaseModel, Field

class RewrittenQueries(BaseModel):
    """Rewritten queries based on the user's original question."""
    rewritten_queries: List[str] = Field(
        description="A list of rewritten versions of the user's query. Each rewritten version is rewritten differently, rephrased and potentially uses synonyms."
    )

rewritten_query_llm = llm.with_structured_output(RewrittenQueries)
rewritten_query_system_prompt = """You are an analyst in charge of taking a user's question as input, and reframing and rewriting it in different ways.\n
Your goal is to change the phrasing of the question, while making sure that the intent and meaning of the question is the same.\n
Return a list of rewritten_queries. The number will be specified by the user."""
rewritten_query_prompt = "Here is the user's question: \n\n {question}. Return {num_rewrites} queries."

def generate_rewritten_queries(state):
    """
    Generates rewritten versions of the original user query

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates rewritten_queries key with a list of rewritten queries
    """
    print("---GENERATING REWRITTEN VERSIONS OF THE USER'S QUERY---")
    question = state["question"]
    num_rewrites = 3
    rewritten_query_prompt_formatted = rewritten_query_prompt.format(question=question, num_rewrites=num_rewrites)
    response = rewritten_query_llm.invoke(
        [SystemMessage(content=rewritten_query_system_prompt)] + [HumanMessage(content=rewritten_query_prompt_formatted)]
    )
    rewritten_queries = response.rewritten_queries

    return {"rewritten_queries": rewritten_queries}

class SampleAnswer(BaseModel):
    """Sample answer for an input question."""
    sample_answer: str = Field(
        description="A concise example answer for a question. This shouldn't exceed three sentences in length."
    )

sample_answer_llm = llm.with_structured_output(SampleAnswer)
sample_answer_system_prompt = """You are a novice in charge of taking a user's question as input, and generating a sample answer for it.\n
This sample answer should contain words that would likely be in a real answer, but is not grounded in any factual documents, the way a real answer would be."""
sample_answer_prompt = "Here is the user's question: \n\n {question}."

def retrieve_documents(state):
    """
    Retrieve documents

    Args:
        state (dict): A dictionary containing a question

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE DOCUMENTS---")
    question = state["question"]
    sample_answer_prompt_formatted = sample_answer_prompt.format(question=question)
    response = sample_answer_llm.invoke(
        [SystemMessage(content=sample_answer_system_prompt)] + [HumanMessage(content=sample_answer_prompt_formatted)]
    )
    sample_answer = response.sample_answer
    documents = retriever.invoke(f"{question}: {sample_answer}")    # Now we use our question and sample answer
    return {"documents": documents}

def continue_to_retrieval_nodes(state: GraphState):
    edges_to_create = []
    # Add original question
    edges_to_create.append(Send("retrieve_documents", {"question": state["question"]}))
    # Add rewritten queries
    for rewritten_query in state["rewritten_queries"]:
        edges_to_create.append(Send("retrieve_documents", {"question": rewritten_query}))
    return edges_to_create

def generate_response(state: GraphState):
    """
    Generate response

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE RESPONSE---")
    question = state["question"]
    documents = state["documents"]
    attempted_generations = state.get("attempted_generations", 0)   # By default we set attempted_generations to 0 if it doesn't exist yet
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    
    # RAG generation
    rag_prompt_formatted = RAG_PROMPT.format(context=formatted_docs, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "attempted_generations": attempted_generations + 1   # In our state update, we increment attempted_generations
    }

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

grade_documents_llm = llm.with_structured_output(GradeDocuments)
grade_documents_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_documents_prompt = "Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}"

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # -- New logic to deduplicate documents our queries --
    doc_counter = Counter(doc.page_content for doc in documents)
    most_common_contents = doc_counter.most_common(5)
    top_documents = []
    for content, _ in most_common_contents:
        for d in documents:
            if d.page_content == content:
                top_documents.append(d)
                break

    # Score each one of our five most common documents
    filtered_docs = []
    for d in top_documents:
        grade_documents_prompt_formatted = grade_documents_prompt.format(document=d.page_content, question=question)
        score = grade_documents_llm.invoke(
            [SystemMessage(content=grade_documents_system_prompt)] + [HumanMessage(content=grade_documents_prompt_formatted)]
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": {"type": "overwrite", "documents": filtered_docs}, "question": question}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or to terminate execution.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, END---"
        )
        return "none relevant"    # same as END
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "some relevant"
    
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

grade_hallucinations_llm = llm.with_structured_output(GradeHallucinations)
grade_hallucinations_system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
grade_hallucinations_prompt = "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"

def grade_hallucinations(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

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
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "supported"
    elif attempted_generations >= ATTEMPTED_GENERATION_MAX:    # New condition!
        print("---DECISION: TOO MANY ATTEMPTS, GIVE UP---")
        raise RuntimeError("Too many attempted generations with hallucinations, giving up.")
        # return "give up"    # Note: We could also do this to silently fail
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

graph_builder = StateGraph(GraphState, input=InputState, output=OutputState)
graph_builder.add_node("generate_rewritten_queries", generate_rewritten_queries)
graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("generate_response", generate_response)
graph_builder.add_node("grade_documents", grade_documents)

graph_builder.add_edge(START, "generate_rewritten_queries")
graph_builder.add_conditional_edges(
    "generate_rewritten_queries",
    continue_to_retrieval_nodes,
    ["retrieve_documents"]
)
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
        "supported": END,
        "not supported": "generate_response"
    })

graph = graph_builder.compile()
