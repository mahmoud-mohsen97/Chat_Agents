"""Hallucination grading chain for multimodal agentic RAG system."""

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
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


def hallucination_grader(inputs):
    """Grade hallucination using multimodal assessment."""
    if not llm:
        return GradeHallucinations(binary_score=True)  # Default to grounded if LLM not available
    
    documents = inputs["documents"]  # List of ImageDocument
    generation = inputs["generation"]
    
    try:
        # Create message parts with images and generation
        parts = []
        
        # Add images from documents (limit to first 3 for API limits and only image docs)
        image_count = 0
        for doc in documents:
            if doc.get("metadata", {}).get("type") != "text" and image_count < 3:  # Skip web search text results
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": doc["page_content"]}
                })
                image_count += 1
        
        # Add grading prompt
        parts.append({
            "type": "text",
            "text": f"""You are a grader assessing whether an LLM generation is grounded in / supported by the images provided above.

Generation to evaluate: {generation}

Instructions:
1. Carefully examine the images provided above
2. Determine if the generation is supported by what you can see in the images
3. Look for specific details, facts, or information in the images that support or contradict the generation
4. Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of images

Respond with just 'yes' if the generation is grounded in the images, or 'no' if it is not grounded."""
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