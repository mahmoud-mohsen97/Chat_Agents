from dotenv import load_dotenv
from graph import app

load_dotenv()

def run_agentic_rag(question: str):
    """Run the agentic RAG system with a question."""
    try:
        inputs = {"question": question}
        
        print(f"\n{'='*50}")
        print(f"QUESTION: {question}")
        print(f"{'='*50}")
        
        # Run the graph
        result = app.invoke(inputs)
        
        print(f"\n{'='*50}")
        print("FINAL ANSWER:")
        print(f"{'='*50}")
        print(result.get("generation", "No answer generated"))
        print(f"{'='*50}\n")
        
        return result
        
    except Exception as e:
        print(f"Error running agentic RAG: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test questions
    test_questions = [
        "What are the company did this person work for?",
        "What is his education?",
        "list the name of all of his projects?",
    ]
    
    for question in test_questions:
        result = run_agentic_rag(question)
        print("\n" + "-"*80 + "\n")
