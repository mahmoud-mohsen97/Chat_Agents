"""Answer grading chain for agentic RAG system."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class GradeAnswer(BaseModel):
    """Binary score for whether an answer addresses the question."""
    
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
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

if llm:
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
         Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
else:
    answer_grader = None 