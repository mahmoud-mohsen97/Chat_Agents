"""Router chain for agentic RAG system."""

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# Initialize LLM with error handling
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    structured_llm_router = llm.with_structured_output(RouteQuery)
except Exception as e:
    print(f"Warning: Could not initialize Google LLM: {e}")
    llm = None
    structured_llm_router = None

if llm and structured_llm_router:
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains document images (like PDFs, CVs, reports) with visual and textual content.
    Use the vectorstore for questions about documents, personal information, qualifications, experience, skills, or any content that would typically be found in documents.
    Use web-search for general knowledge questions, current events, or information not typically found in personal/professional documents."""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
else:
    # Default router that always goes to vectorstore
    def default_router(inputs):
        return RouteQuery(datasource="vectorstore")
    
    question_router = default_router 