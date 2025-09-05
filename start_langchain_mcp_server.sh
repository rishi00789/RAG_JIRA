#!/bin/bash

# LangChain-powered JIRA RAG MCP Server Startup Script
# This script starts the LangChain-powered MCP server with proper environment setup

echo "🚀 Starting LangChain-powered JIRA RAG MCP Server..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required environment variables are set
echo "🔍 Checking environment variables..."
required_vars=("JIRA_BASE_URL" "JIRA_USERNAME" "JIRA_API_TOKEN")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "   $var"
    done
    echo ""
    echo "Please set these variables in your .env file or environment:"
    echo "   JIRA_BASE_URL=https://your-domain.atlassian.net"
    echo "   JIRA_USERNAME=your-email@domain.com"
    echo "   JIRA_API_TOKEN=your-api-token"
    echo "   HF_TOKEN=your-huggingface-token (optional)"
    exit 1
fi

echo "✅ Environment variables configured"

# Check if Milvus is running
echo "🔍 Checking Milvus connection..."
if ! curl -s http://localhost:19530/health > /dev/null 2>&1; then
    echo "⚠️  Milvus is not running on localhost:19530"
    echo "   Please start Milvus first:"
    echo "   docker run -p 19530:19530 -d milvusdb/milvus:latest"
    echo ""
    echo "   Or if using docker-compose:"
    echo "   docker-compose up -d"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Milvus is running"
fi

# Install/upgrade dependencies if needed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Check if LangChain components are available
echo "🧪 Testing LangChain components..."
python3 -c "
try:
    from langchain_jira_loader import JiraDocumentLoader
    from langchain_milvus_store import MilvusVectorStore
    from langchain_rag_chain import create_jira_rag_chain
    print('✅ LangChain components available')
except ImportError as e:
    print(f'❌ LangChain components not available: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ LangChain components test failed"
    exit 1
fi

# Create necessary directories
mkdir -p logs
mkdir -p data

# Set up logging
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"

echo ""
echo "🎯 Starting LangChain JIRA RAG MCP Server..."
echo "   Server will be available at: http://127.0.0.1:8003/mcp"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the server
cd backend
python3 langchain_mcp_server.py
