# 🤖 Chat Agents - Dual AI Agent System

A powerful dual-agent system combining **Agentic RAG** for document processing and **Deep Researcher** for web-based research, built with FastAPI and Streamlit.

## 🏗️ System Architecture

![Project Architecture](https://raw.githubusercontent.com/user/repo/main/docs/project-architecture.png)

The system consists of three main components:
- **Streamlit Frontend**: Interactive user interface
- **FastAPI Backend**: RESTful API with dual agent services  
- **LangGraph Agents**: AI-powered document processing and research

## 🎯 Features

### 📚 Agentic RAG Agent
- **Multimodal PDF Processing**: Extract text and images from PDFs
- **Intelligent Document Retrieval**: Vector-based similarity search
- **Smart Routing**: Automatically determines if questions are document-related
- **Hallucination Prevention**: Built-in answer validation and grading
- **Web Search Fallback**: Searches the web when documents don't contain relevant information

### 🔍 Deep Researcher Agent  
- **Autonomous Research**: Generate comprehensive reports on any topic
- **Multi-Query Search**: Creates 4 specialized search queries per research task
- **Persona-Based Planning**: Develops research personas for targeted investigations
- **Web Search Integration**: Powered by Tavily API for real-time information
- **Structured Output**: Generates professional markdown reports with references

## 🔧 Agent Architectures

### Agentic RAG Workflow
![Agentic RAG Architecture](https://raw.githubusercontent.com/user/repo/main/docs/agentic-rag-architecture.png)

**Processing Flow:**
1. **Question Routing**: Determines if query relates to uploaded documents
2. **Document Retrieval**: Searches vector database for relevant content  
3. **Relevance Grading**: Validates retrieved documents against query
4. **Answer Generation**: Creates responses using relevant context
5. **Hallucination Check**: Verifies answer accuracy against source material
6. **Web Search Fallback**: Searches online when documents are insufficient

### Deep Researcher Workflow
![Deep Researcher Architecture](https://raw.githubusercontent.com/user/repo/main/docs/researcher-architecture.png)

**Research Process:**
1. **Task**: Accepts user research query
2. **Planner**: Develops specialized researcher persona and strategy
3. **Researcher**: Executes 4 targeted web searches using different angles
4. **Publisher**: Synthesizes findings into comprehensive markdown report

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- API Keys: Openai, Google Gemini, Cohere, and Tavily API Key

### Setup

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd Chat_Agents
   ```

2. **Configure Environment**
   ```bash
   cp env.template .env
   # Edit .env with your API keys:
   # OPENAI_API_KEY=your_openai_api_key_here
   # GOOGLE_API_KEY=your_google_api_key_here
   # COHERE_API_KEY=your_cohere_api_key_here
   # TAVILY_API_KEY=your_tavily_api_key_here
   ```

3. **Launch Application**
   ```bash
   ./start.sh
   ```

4. **Access Services**
   - **Frontend**: http://localhost:8501
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

## 📡 API Endpoints

### Agentic RAG Endpoints
```http
POST   /api/agentic-rag/upload-pdf     # Upload PDF documents
POST   /api/agentic-rag/query          # Ask questions about documents
GET    /api/agentic-rag/status         # Service health check
```

### Deep Researcher Endpoints
```http
POST   /api/researcher/generate-report   # Generate research report
GET    /api/researcher/reports           # List saved reports
GET    /api/researcher/reports/{id}      # Get specific report
GET    /api/researcher/reports/{id}/download  # Download report
GET    /api/researcher/status            # Service health check
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance async web framework
- **LangGraph**: AI agent orchestration framework
- **LangChain**: LLM application development framework
- **Qdrant**: Vector database for document embeddings
- **Google AI**: Language model provider
- **Tavily**: Web search API

### Frontend
- **Streamlit**: Interactive web application framework
- **Python**: Core programming language

### Infrastructure
- **Docker**: Containerization platform
- **Docker Compose**: Multi-container orchestration
- **uv**: Fast Python package management

## 📁 Project Structure

```
Chat_Agents/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── agents/            # AI agent implementations
│   │   │   ├── agentic_rag/   # Document processing agent
│   │   │   └── researcher/    # Research agent
│   │   ├── models/            # Pydantic schemas
│   │   ├── utils/             # Utility functions
│   │   └── main.py            # FastAPI application
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                   # Streamlit frontend
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── storage/                    # Runtime data
│   ├── reports/               # Generated research reports
│   ├── uploads/               # Uploaded PDF files
│   └── qdrant_data/          # Vector database storage
├── docker-compose.yml         # Container orchestration
├── start.sh                   # Application launcher
└── README.md                  # Documentation
```

## 🎮 Usage Guide

### Document Processing (Agentic RAG)
1. Navigate to the **Agentic RAG** tab
2. Upload PDF documents using the file uploader
3. Wait for processing completion
4. Ask questions about your documents
5. Receive intelligent answers with source references

### Research Generation (Deep Researcher)
1. Switch to the **Deep Researcher** tab
2. Choose **"Generate New Report"**
3. Enter your research topic
4. Click **"Generate Report"**
5. Monitor real-time progress
6. View and download completed reports

### Report Management
- **View Reports**: Browse saved research reports
- **Download**: Export reports as markdown files
- **Metadata**: See word count, search results, and timestamps

## ⚙️ Configuration

### Environment Variables
```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333
BACKEND_URL=http://backend:8000
```

### Docker Configuration
- **Backend Port**: 8000
- **Frontend Port**: 8501  
- **Qdrant Port**: 6333
- **Shared Volumes**: Storage, Qdrant data

## 🔍 Key Features in Detail

### Intelligent Document Processing
- **Multi-format Support**: PDF text and image extraction
- **Semantic Search**: Vector-based document retrieval
- **Context Preservation**: Maintains document structure and relationships
- **Quality Assurance**: Answer validation and hallucination detection

### Advanced Research Capabilities
- **Dynamic Query Generation**: Creates multiple search perspectives
- **Comprehensive Coverage**: 4 targeted searches per research topic
- **Professional Output**: Structured markdown reports with citations
- **Persistent Storage**: Automatic report saving and management

### Production-Ready Architecture
- **Microservices Design**: Separate frontend and backend containers
- **Async Processing**: Non-blocking operations for better performance
- **Error Handling**: Graceful failure management and user feedback
- **Scalable Storage**: Vector database and file system integration

## 🚦 Health Monitoring

Check service status:
```bash
# Backend health
curl http://localhost:8000/health

# Individual service status
curl http://localhost:8000/api/agentic-rag/status
curl http://localhost:8000/api/researcher/status
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request


## 🙏 Acknowledgments

- **LangChain**: For the powerful LLM application framework
- **LangGraph**: For agent orchestration capabilities
- **Streamlit**: For the intuitive frontend framework
- **FastAPI**: For the high-performance backend framework
- **Qdrant**: For vector database functionality
- **Google AI**: For language model services
- **Tavily**: For web search capabilities

---

**Built by me with ❤️ using modern AI and web technologies**

For support, questions, or feature requests, please open an issue in the repository.
