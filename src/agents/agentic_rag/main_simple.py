#!/usr/bin/env python3
"""
Simple main script for multimodal agentic RAG using direct Google Generative AI.
This bypasses the LangChain authentication issues.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from ingestion import get_retriever
import base64
from io import BytesIO
from PIL import Image

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def data_url_to_pil_image(data_url: str) -> Image.Image:
    """Convert data URL back to PIL Image for Gemini."""
    header, data = data_url.split(',', 1)
    image_data = base64.b64decode(data)
    return Image.open(BytesIO(image_data))

def simple_agentic_rag(question: str, model_name="gemini-1.5-flash"):
    """
    Simple agentic RAG flow:
    1. Retrieve relevant images
    2. Grade relevance (simple scoring)
    3. Generate answer with Gemini
    4. Return result
    """
    print(f"\n{'='*60}")
    print(f"🤔 QUESTION: {question}")
    print(f"{'='*60}")
    
    # Initialize model and retriever
    model = genai.GenerativeModel(model_name)
    retriever = get_retriever()
    
    print("🔍 STEP 1: Retrieving relevant documents...")
    
    # Retrieve documents
    docs = retriever.invoke(question)
    print(f"   Retrieved {len(docs)} documents")
    
    # Grade relevance (simple scoring based on similarity)
    relevant_docs = []
    for doc in docs:
        score = doc.metadata.get('score', 0)
        page = doc.metadata.get('page', 'unknown')
        print(f"   📄 Page {page}: Score = {score:.3f}")
        
        # Simple grading: keep documents with score > 0.25
        if score > 0.25:
            relevant_docs.append(doc)
            print(f"      ✅ Relevant (score > 0.25)")
        else:
            print(f"      ❌ Not relevant (score <= 0.25)")
    
    print(f"\n📊 STEP 2: Document grading complete")
    print(f"   Relevant documents: {len(relevant_docs)}/{len(docs)}")
    
    if not relevant_docs:
        return "❌ No relevant documents found. Try rephrasing your question."
    
    print(f"\n🤖 STEP 3: Generating answer with Gemini Flash...")
    
    try:
        # Prepare images and prompt
        images = []
        for doc in relevant_docs:
            image = data_url_to_pil_image(doc.page_content)
            images.append(image)
        
        # Create comprehensive prompt
        prompt = f"""You are an expert document analyst. Based on the provided document images, please answer the following question comprehensively:

Question: {question}

Instructions:
1. Analyze the visual content of the document images carefully
2. Extract relevant information that directly answers the question
3. Provide specific details, requirements, or information visible in the documents
4. Structure your answer clearly with bullet points or sections if appropriate
5. If you can't find relevant information, clearly state that

Please provide a detailed and accurate answer based on what you can see in the images."""

        # Generate response
        content = images + [prompt]
        response = model.generate_content(content)
        
        print(f"✅ STEP 4: Answer generated successfully")
        
        # Simple hallucination check (basic length and content validation)
        answer = response.text
        if len(answer) < 50:
            print("⚠️  Warning: Answer seems too short, might not be comprehensive")
        
        print(f"\n🎯 FINAL ANSWER:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
        
        return answer
        
    except Exception as e:
        error_msg = f"❌ Error generating answer: {e}"
        print(error_msg)
        return error_msg

def interactive_mode():
    """Run the system in interactive mode."""
    print("🚀 Multimodal Agentic RAG System - Interactive Mode")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            question = input("💭 Ask a question about the document: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("❓ Please enter a question.")
                continue
            
            # Process the question
            simple_agentic_rag(question)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def run_examples():
    """Run some example questions."""
    examples = [
        "What are the required skills for this position?",
        "What experience is needed for this job?",
        "What technologies or programming languages should I know?",
        "What are the main responsibilities?",
    ]
    
    print("🧪 Running example questions...\n")
    
    for question in examples:
        simple_agentic_rag(question)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    print("🎯 Multimodal Agentic RAG System")
    print("Choose an option:")
    print("1. Run example questions")
    print("2. Interactive mode")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            run_examples()
            break
        elif choice == "2":
            interactive_mode()
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.") 