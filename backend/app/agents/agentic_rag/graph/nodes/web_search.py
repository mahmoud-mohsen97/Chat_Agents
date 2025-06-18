"""Web search node for agentic RAG system."""

from typing import Any, Dict
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from ..state import GraphState, ImageDocument

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


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


if __name__ == "__main__":
    # Test web search
    test_state = {"question": "agent memory", "documents": []}
    result = web_search(test_state)
    print(f"Web search result: {result}") 