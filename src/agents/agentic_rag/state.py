from typing import List, TypedDict, Dict, Any


class ImageDocument(TypedDict):
    """Represents an image document with metadata."""
    page_content: str  # base64 data URL
    page_number: int
    metadata: Dict[str, Any]


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of image documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[ImageDocument]
