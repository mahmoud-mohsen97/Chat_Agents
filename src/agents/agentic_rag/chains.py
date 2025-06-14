"""
Agentic RAG Chains Module

This module contains various chains and graders for an agentic RAG system:
- Answer grading
- Generation chain
- Hallucination detection
- Document relevance grading
- Query routing
"""

from typing import Literal, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel, Field
from state import ImageDocument


# =============================================================================
# LLM Configuration
# =============================================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class GradeAnswer(BaseModel):
    """Binary score for whether an answer addresses the question."""
    
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# =============================================================================
# Answer Grading Chain
# =============================================================================

structured_llm_grader = llm.with_structured_output(GradeAnswer)

answer_system_prompt = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader


# =============================================================================
# Multimodal Generation Chain
# =============================================================================

def create_multimodal_generation_chain():
    """Create a multimodal generation chain that handles images and text."""
    
    def generate_multimodal_answer(inputs):
        question = inputs["question"]
        documents = inputs["context"]  # List of ImageDocument
        
        # Create message parts with images and question
        parts = []
        
        # Add images from retrieved documents
        for doc in documents:
            parts.append({
                "type": "image_url",
                "image_url": {"url": doc["page_content"]}
            })
        
        # Add the question
        parts.append({
            "type": "text", 
            "text": f"""Based on the images provided, please answer the following question: {question}
            
            Provide a comprehensive answer based on what you can see in the images. If the information is not clearly visible in the images, please state that clearly."""
        })
        
        message = HumanMessage(content=parts)
        response = llm.invoke([message])
        return response.content
    
    return generate_multimodal_answer

generation_chain = create_multimodal_generation_chain()


# =============================================================================
# Hallucination Grading Chain (Modified for Multimodal)
# =============================================================================

hallucination_llm_grader = llm.with_structured_output(GradeHallucinations)

def create_multimodal_hallucination_grader():
    """Create hallucination grader that works with images."""
    
    def grade_hallucination(inputs):
        documents = inputs["documents"]  # List of ImageDocument
        generation = inputs["generation"]
        
        try:
            # Create message parts with images and generation
            parts = []
            
            # Add images from documents (limit to first 3 for API limits)
            for i, doc in enumerate(documents[:3]):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": doc["page_content"]}
                })
            
            # Add grading prompt
            parts.append({
                "type": "text",
                "text": f"""You are a grader assessing whether an LLM generation is grounded in / supported by the images provided above.
                
                Generation to evaluate: {generation}
                
                Look at the images and determine if the generation is supported by what you can see in the images.
                Respond with a simple 'yes' if grounded or 'no' if not grounded."""
            })
            
            message = HumanMessage(content=parts)
            response = llm.invoke([message])
            
            # Parse the response
            response_text = response.content.lower().strip()
            binary_score = "yes" in response_text or "true" in response_text
            
            return GradeHallucinations(binary_score=binary_score)
            
        except Exception as e:
            print(f"Error in hallucination grading: {e}")
            # Default to grounded if error occurs
            return GradeHallucinations(binary_score=True)
    
    return grade_hallucination

hallucination_grader = create_multimodal_hallucination_grader()


# =============================================================================
# Document Relevance Grading Chain (Modified for Multimodal)
# =============================================================================

document_llm_grader = llm.with_structured_output(GradeDocuments)

def create_multimodal_document_grader():
    """Create document grader that works with images."""
    
    def grade_document_relevance(inputs):
        document = inputs["document"]  # Single ImageDocument
        question = inputs["question"]
        
        try:
            # Create message parts with image and question
            parts = [
                {
                    "type": "image_url",
                    "image_url": {"url": document["page_content"]}
                },
                {
                    "type": "text",
                    "text": f"""You are a grader assessing relevance of the image above to a user question.
                    
                    User question: {question}
                    
                    Look at the image and determine if it contains visual information or content related to the question.
                    Respond with a simple 'yes' if relevant or 'no' if not relevant."""
                }
            ]
            
            message = HumanMessage(content=parts)
            response = llm.invoke([message])
            
            # Parse the response
            response_text = response.content.lower().strip()
            binary_score = "yes" if ("yes" in response_text or "relevant" in response_text) else "no"
            
            return GradeDocuments(binary_score=binary_score)
            
        except Exception as e:
            print(f"Error in document grading: {e}")
            # Default to relevant if error occurs
            return GradeDocuments(binary_score="yes")
    
    return grade_document_relevance

retrieval_grader = create_multimodal_document_grader()


# =============================================================================
# Query Routing Chain
# =============================================================================

structured_llm_router = llm.with_structured_output(RouteQuery)

routing_system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains visual documents (images) related to cv of a person.
Use the vectorstore for questions about the person's education, work experience, skills, qualifications, or anything that might be documented in cv.
For general questions not related to the specific person or technical requirements, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", routing_system_prompt),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

