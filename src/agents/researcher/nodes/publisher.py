"""Publisher Node Module

Synthesizes search results into a comprehensive markdown report.
"""

from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:
    from ..state import GraphState
    from ..prompts import get_publisher_prompt
except ImportError:
    import sys
    sys.path.append('..')
    from state import GraphState
    from prompts import get_publisher_prompt

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,  # Slightly higher for more creative synthesis
)

def publisher_node(state: GraphState) -> Dict[str, Any]:
    """
    Synthesize search results into a markdown report.
    
    Args:
        state: Current graph state with user_query and search_results
        
    Returns:
        Updated state with markdown_answer
    """
    print("---PUBLISHER NODE---")
    user_query = state["user_query"]
    search_results = state["search_results"]
    
    # Create prompt for report generation
    prompt_template = ChatPromptTemplate.from_template(
        get_publisher_prompt(user_query, search_results)
    )
    chain = prompt_template | llm | StrOutputParser()
    
    # Generate the markdown report
    markdown_answer = chain.invoke({
        "query": user_query,
        "results": search_results
    })
    
    print(f"Generated markdown report ({len(markdown_answer)} characters)")
    
    return {"markdown_answer": markdown_answer} 