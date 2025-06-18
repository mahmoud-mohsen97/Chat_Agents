"""Generation chain for multimodal agentic RAG system."""

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM with error handling
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
except Exception as e:
    print(f"Warning: Could not initialize Google LLM: {e}")
    llm = None


def generation_chain(inputs):
    """Generate answer using multimodal chain with retrieved images."""
    if not llm:
        return "Error: LLM not initialized. Please check your API key configuration."
    
    question = inputs["question"]
    documents = inputs["context"]  # List of ImageDocument
    
    # Create message parts with images and question
    parts = []
    
    # Add images from retrieved documents (only image documents, not web search text)
    for doc in documents:
        if doc.get("metadata", {}).get("type") != "text":  # Skip web search text results
            parts.append({
                "type": "image_url",
                "image_url": {"url": doc["page_content"]}
            })
    
    # Add the question with comprehensive prompt
    prompt_text = f"""Based on the images provided above, please answer the following question: {question}

Instructions:
1. Carefully examine all the images provided
2. Provide a comprehensive and accurate answer based on what you can observe in the images
3. If the information needed to answer the question is not clearly visible in the images, state this clearly
4. Be specific and reference details you can see in the images when possible
5. If multiple images are provided, consider information from all of them in your answer

Question: {question}

Answer:"""
    
    parts.append({
        "type": "text", 
        "text": prompt_text
    })
    
    message = HumanMessage(content=parts)
    response = llm.invoke([message])
    return response.content 