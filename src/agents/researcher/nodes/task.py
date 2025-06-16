"""Task Node Module

Handles the initial task input and query processing.
"""

from typing import Dict, Any
try:
    from ..state import GraphState
except ImportError:
    import sys
    sys.path.append('..')
    from state import GraphState


def task_node(state: GraphState) -> Dict[str, Any]:
    """
    Initial task processing node that takes user query as input.
    
    Args:
        state: Current graph state containing user_query
        
    Returns:
        Updated state with user_query maintained
    """
    print("---TASK NODE---")
    user_query = state["user_query"]
    print(f"Processing query: {user_query}")
    
    return {"user_query": user_query} 