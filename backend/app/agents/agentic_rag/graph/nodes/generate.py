"""Generate node for agentic RAG system."""

from typing import Any, Dict
from ..chains.generation import generation_chain
from ..state import GraphState


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