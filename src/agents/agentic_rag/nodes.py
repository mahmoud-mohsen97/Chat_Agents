from typing import Any, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch

from state import GraphState, ImageDocument
from ingestion import get_retriever
from chains import retrieval_grader, generation_chain, hallucination_grader

load_dotenv()
web_search_tool = TavilySearch(max_results=3)

def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant image documents from the vector store."""
    print("---RETRIEVE---")
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    
    # Convert retrieved documents to ImageDocument format
    image_docs = []
    for doc in documents:
        image_doc = ImageDocument(
            page_content=doc.page_content,
            page_number=doc.metadata.get("page", 0),
            metadata=doc.metadata
        )
        image_docs.append(image_doc)
    
    return {"documents": image_docs, "question": question}

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved image documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for doc in documents:
        try:
            score = retrieval_grader({"question": question, "document": doc})
            grade = score.binary_score
            if grade.lower() == "yes":
                print(f"---GRADE: DOCUMENT PAGE {doc['page_number']} RELEVANT---")
                filtered_docs.append(doc)
            else:
                print(f"---GRADE: DOCUMENT PAGE {doc['page_number']} NOT RELEVANT---")
                web_search = True
                continue
        except Exception as e:
            print(f"---ERROR GRADING DOCUMENT PAGE {doc['page_number']}: {e}---")
            # Keep the document if grading fails
            filtered_docs.append(doc)
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generate(state: GraphState) -> Dict[str, Any]:
    """Generate answer using multimodal chain with retrieved images."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    try:
        generation = generation_chain({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    except Exception as e:
        print(f"---ERROR IN GENERATION: {e}---")
        # Fallback generation
        fallback_generation = f"I apologize, but I encountered an error while processing the images to answer your question: {question}. Please try rephrasing your question or check if the images are properly formatted."
        return {"documents": documents, "question": question, "generation": fallback_generation}

def web_search(state: GraphState) -> Dict[str, Any]:
    """Perform web search and convert results to ImageDocument format."""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
        
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    
    # Convert web search result to ImageDocument format
    # Note: Web search results are text, so we store them as text content
    # with a special marker indicating they are text-based
    web_doc = ImageDocument(
        page_content=joined_tavily_result,  # This is text, not image
        page_number=-1,  # Special marker for web search results
        metadata={"source": "web_search", "type": "text"}
    )
    
    if documents:
        documents.append(web_doc)
    else:
        documents = [web_doc]
    
    return {"documents": documents, "question": question}

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
            # This doesn't require images, so we can use the original approach
            from chains import answer_grader
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

if __name__ == "__main__":
    # Test web search
    test_state = {"question": "agent memory", "documents": []}
    result = web_search(test_state)
    print(f"Web search result: {result}")
