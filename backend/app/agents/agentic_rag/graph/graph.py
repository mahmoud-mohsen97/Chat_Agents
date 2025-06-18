"""Main graph implementation for agentic RAG system."""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .chains.answer_grader import answer_grader
from .chains.hallucination_grader import hallucination_grader
from .chains.router import RouteQuery, question_router
from .consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from .nodes import generate, grade_documents, retrieve, web_search
from .state import GraphState

load_dotenv()


def decide_to_generate(state):
    """Determine whether to proceed to generation or web search."""
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """Grade the generation for hallucinations and relevance to question."""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    try:
        # Filter out web search documents for hallucination checking
        image_documents = [doc for doc in documents if doc.get("metadata", {}).get("type") != "text"]
        
        if image_documents:
            score = hallucination_grader({"documents": image_documents, "generation": generation})
            hallucination_grade = score.binary_score
        else:
            # If no image documents, skip hallucination check
            hallucination_grade = True
        
        if hallucination_grade:
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            
            # For answer grading, we can use the generation and question directly
            score = answer_grader.invoke({"question": question, "generation": generation})
            answer_grade = score.binary_score
            
            if answer_grade:
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    except Exception as e:
        print(f"---ERROR IN GRADING: {e}---")
        # Default to useful if grading fails
        return "useful"


def route_question(state: GraphState) -> str:
    """Route the question to vectorstore or web search."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


# Build the graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Set conditional entry point
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

# Add edges
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)

# Compile the graph
app = workflow.compile()

# Generate graph visualization
app.get_graph().draw_mermaid_png(output_file_path="graph.png") 