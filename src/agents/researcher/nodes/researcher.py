"""Researcher Node Module

Generates search queries and executes web searches using Tavily.
"""

import ast
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

try:
    from ..state import GraphState
    from ..prompts import get_search_queries_prompt
except ImportError:
    import sys
    sys.path.append('..')
    from state import GraphState
    from prompts import get_search_queries_prompt

load_dotenv()

# Initialize LLM and search tool
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

tavily_search = TavilySearch(max_results=3)

def researcher_node(state: GraphState) -> Dict[str, Any]:
    """
    Generate 4 search queries and execute them with Tavily.
    
    Args:
        state: Current graph state with user_query and persona_prompt
        
    Returns:
        Updated state with search_results
    """
    print("---RESEARCHER NODE---")
    user_query = state["user_query"]
    
    # Generate 4 search queries
    prompt_template = ChatPromptTemplate.from_template(get_search_queries_prompt(user_query))
    chain = prompt_template | llm | StrOutputParser()
    
    queries_response = chain.invoke({"query": user_query})
    print(f"LLM response for queries: {queries_response[:200]}...")
    
    # Parse the response to extract the list of queries
    try:
        # First try to parse as a Python list
        search_queries = ast.literal_eval(queries_response)
        if not isinstance(search_queries, list) or len(search_queries) != 4:
            raise ValueError("Invalid list format")
    except:
        # Fallback: split by lines and clean up
        lines = queries_response.strip().split('\n')
        search_queries = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-') and not line.startswith('*'):
                # Remove leading numbers, bullets, quotes
                line = line.lstrip('1234567890.- ').strip('"\'')
                if line:
                    search_queries.append(line)
        
        # If we still don't have 4 queries, create variations manually
        if len(search_queries) < 4:
            base_variations = [
                f"{user_query} recent developments",
                f"{user_query} current status 2024", 
                f"{user_query} latest news analysis",
                f"{user_query} expert opinions trends"
            ]
            search_queries = base_variations[:4]
    
    print(f"Generated {len(search_queries)} search queries")
    
    # Execute each search query
    all_results = []
    for i, query in enumerate(search_queries):
        try:
            print(f"Searching: {query}")
            results = tavily_search.invoke({"query": query})
            
            # Extract content from results
            for result in results:
                if isinstance(result, dict) and "content" in result:
                    all_results.append(result["content"])
                elif isinstance(result, str):
                    all_results.append(result)
                    
        except Exception as e:
            print(f"Search error for query {i}: {e}")
            all_results.append(f"Search failed for: {query}")
    
    print(f"Collected {len(all_results)} search results")
    return {"search_results": all_results} 