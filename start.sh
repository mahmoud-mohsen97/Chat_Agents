#!/bin/bash

# AI Agents Production Startup Script

echo "🚀 Starting AI Agents Production System..."
echo "=========================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your API keys:"
    echo ""
    echo "OPENAI_API_KEY=your_openai_api_key"
    echo "GOOGLE_API_KEY=your_google_api_key"
    echo "COHERE_API_KEY=your_cohere_api_key"
    echo "TAVILY_API_KEY=your_tavily_api_key"
    echo ""
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed!"
    echo "Please install Docker Compose and try again."
    exit 1
fi

# Create necessary directories
echo "📁 Creating storage directories..."
mkdir -p storage/uploads
mkdir -p storage/reports
mkdir -p storage/qdrant_data

# Pull latest images and build
echo "🐳 Building and starting services..."
docker-compose down
docker-compose pull
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check Qdrant
echo "Checking Qdrant database..."
if curl -s http://localhost:6333/health > /dev/null; then
    echo "✅ Qdrant is running"
else
    echo "⚠️  Qdrant may still be starting up..."
fi

# Check Backend
echo "Checking Backend API..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend API is running"
else
    echo "⚠️  Backend API may still be starting up..."
fi

# Check Frontend
echo "Checking Frontend..."
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Frontend is running"
else
    echo "⚠️  Frontend may still be starting up..."
fi

echo ""
echo "🎉 AI Agents System is starting up!"
echo "=========================================="
echo "🌐 Frontend UI:    http://localhost:8501"
echo "🔧 Backend API:    http://localhost:8000"
echo "📖 API Docs:       http://localhost:8000/docs"
echo "🗄️  Qdrant DB:     http://localhost:6333/dashboard"
echo ""
echo "📝 To view logs: docker-compose logs -f"
echo "🛑 To stop:      docker-compose down"
echo ""
echo "Happy chatting with your AI agents! 🤖" 