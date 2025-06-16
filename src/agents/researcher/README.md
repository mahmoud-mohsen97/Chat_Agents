# Research Agent

A minimal, self-contained research agent that follows a **Task → Planner → Researcher → Publisher** workflow.

## Directory Structure

```
src/agents/researcher/
├── __init__.py            # export build_graph()
├── graph.py               # defines build_graph()
├── nodes/
│   ├── task.py            # Task node wrapper (takes user query)
│   ├── planner.py         # creates agent persona / instructions
│   ├── researcher.py      # 4× Tavily searches + gather results
│   └── publisher.py       # compiles markdown answer
├── prompts.py             # reusable prompt templates
├── state.py               # GraphState definition
└── main.py                # usage example
```

## Usage

### Basic Usage

```python
from src.agents.researcher import build_graph

# Build the research agent
agent = build_graph()

# Run a research query
result = agent.invoke({
    "user_query": "What are the latest trends in AI research?",
    "persona_prompt": "",
    "search_results": [],
    "markdown_answer": ""
})

# Get the final markdown report
print(result["markdown_answer"])
```

### Running the Example

```bash
cd src/agents/researcher
python main.py
```

## Workflow

1. **Task**: Takes user query and passes it downstream
2. **Planner**: Generates researcher persona/instructions based on query  
3. **Researcher**: Creates 4 distinct search queries and executes them with Tavily
4. **Publisher**: Synthesizes search results into structured markdown report

## Requirements

- `langchain-google-genai`
- `langchain-tavily` 
- `langgraph`
- `python-dotenv`

## Environment Variables

Set your API keys in `.env`:

```
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
``` 