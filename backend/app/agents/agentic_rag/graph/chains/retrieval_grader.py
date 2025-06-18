"""Retrieval grading chain for multimodal agentic RAG system."""

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# Initialize LLM with error handling
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
except Exception as e:
    print(f"Warning: Could not initialize Google LLM: {e}")
    llm = None


def retrieval_grader(inputs):
    """Grade document relevance using multimodal assessment."""
    if not llm:
        return GradeDocuments(binary_score="yes")  # Default to relevant if LLM not available
    
    document = inputs["document"]  # Single ImageDocument
    question = inputs["question"]
    
    try:
        # Handle text documents from web search differently
        if document.get("metadata", {}).get("type") == "text":
            # For text documents, use text-based grading
            parts = [{
                "type": "text",
                "text": f"""You are a grader assessing relevance of a retrieved document to a user question.

Document content: {document["page_content"]}

User question: {question}

Instructions:
1. If the document contains keywords or semantic meaning related to the question, grade it as relevant
2. Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question

Respond with just 'yes' if relevant or 'no' if not relevant."""
            }]
        else:
            # For image documents, use multimodal grading
            parts = [
                {
                    "type": "image_url",
                    "image_url": {"url": document["page_content"]}
                },
                {
                    "type": "text",
                    "text": f"""You are a grader assessing relevance of a retrieved image document to a user question.

User question: {question}

Instructions:
1. Carefully examine the image above
2. If the image contains visual information, text, or content related to the question, grade it as relevant
3. Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question

Respond with just 'yes' if relevant or 'no' if not relevant."""
                }
            ]
        
        message = HumanMessage(content=parts)
        response = llm.invoke([message])
        
        # Parse the response
        response_text = response.content.lower().strip()
        binary_score = "yes" if "yes" in response_text else "no"
        
        return GradeDocuments(binary_score=binary_score)
        
    except Exception as e:
        print(f"Error in document grading: {e}")
        # Default to relevant if error occurs
        return GradeDocuments(binary_score="yes") 