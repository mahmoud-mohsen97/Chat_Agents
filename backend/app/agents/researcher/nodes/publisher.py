"""Publisher Node Module

Synthesizes search results into a comprehensive markdown report.
"""

from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..state import GraphState
from ..prompts import get_publisher_prompt

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    temperature=0.3,  # Slightly higher for more creative synthesis
)

def publisher_node(state: GraphState) -> Dict[str, Any]:
    """
    Synthesize search results into a markdown report.
    
    Args:
        state: Current graph state with question and search_results
        
    Returns:
        Updated state with markdown_answer
    """
    print("---PUBLISHER NODE---")
    print(f"🔥 PUBLISHER DEBUG: State keys: {list(state.keys())}")
    print(f"🔥 PUBLISHER DEBUG: State content: {state}")
    
    question = state["question"]
    
    # Check if search_results exists
    if "search_results" not in state:
        print("🔥 PUBLISHER DEBUG: ERROR - search_results not in state!")
        print(f"🔥 PUBLISHER DEBUG: Available keys: {list(state.keys())}")
        raise KeyError("'search_results' not found in state")
    
    search_results = state["search_results"]
    print(f"🔥 PUBLISHER DEBUG: Found {len(search_results)} search results")
    
    # Create prompt for report generation
    prompt_template = ChatPromptTemplate.from_template(
        get_publisher_prompt(question, search_results)
    )
    chain = prompt_template | llm | StrOutputParser()
    
    # Generate the markdown report
    markdown_answer = chain.invoke({
        "query": question,
        "results": search_results
    })
    
    print(f"Generated markdown report ({len(markdown_answer)} characters)")
    
    # Return complete state to ensure proper merging
    result = {
        "question": question,
        "persona_prompt": state.get("persona_prompt", ""),
        "search_results": search_results,
        "markdown_answer": markdown_answer
    }
    print(f"🔥 PUBLISHER DEBUG: Complete state being returned: {list(result.keys())}")
    return result 