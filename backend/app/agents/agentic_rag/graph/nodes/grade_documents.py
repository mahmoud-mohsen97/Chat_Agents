"""Grade documents node for agentic RAG system."""

from typing import Any, Dict
from ..chains.retrieval_grader import retrieval_grader
from ..state import GraphState


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