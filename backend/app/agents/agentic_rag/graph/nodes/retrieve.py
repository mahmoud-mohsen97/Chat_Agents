"""Retrieve node for agentic RAG system."""

from typing import Any, Dict
from ..state import GraphState, ImageDocument
from ...ingestion import get_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant image documents from the vector store."""
    print("---RETRIEVE---")
    question = state["question"]
    
    # Use Docker-compatible Qdrant connection
    import os
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    
    retriever = get_retriever(
        collection="pdf_pages",
        host=qdrant_host,
        port=qdrant_port
    )
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