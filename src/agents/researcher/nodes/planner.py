"""Planner Node Module

Creates agent persona and instructions based on the user query.
"""

from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:
    from ..state import GraphState
    from ..prompts import get_planner_prompt
except ImportError:
    import sys
    sys.path.append('..')
    from state import GraphState
    from prompts import get_planner_prompt

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    Generate researcher persona based on user query.
    
    Args:
        state: Current graph state containing user_query
        
    Returns:
        Updated state with persona_prompt
    """
    print("---PLANNER NODE---")
    user_query = state["user_query"]
    
    # Create prompt for persona generation
    prompt_template = ChatPromptTemplate.from_template(get_planner_prompt(user_query))
    chain = prompt_template | llm | StrOutputParser()
    
    persona_prompt = chain.invoke({"query": user_query})
    print(f"Generated persona: {persona_prompt[:100]}...")
    
    return {"persona_prompt": persona_prompt} 