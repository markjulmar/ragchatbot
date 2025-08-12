import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from pathlib import Path
import os

from config import config
from rag_system import RAGSystem
from ai_generator import AIGenerator
from vector_store import VectorStore
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from session_manager import SessionManager


@pytest.fixture(scope="session")
def test_config():
    """Mock configuration for testing"""
    config_mock = Mock()
    config_mock.CHUNK_SIZE = 800
    config_mock.CHUNK_OVERLAP = 100
    config_mock.CHROMA_PATH = ":memory:"
    config_mock.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config_mock.MAX_RESULTS = 5
    config_mock.ANTHROPIC_API_KEY = "test_key"
    config_mock.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config_mock.MAX_HISTORY = 2
    return config_mock


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for AI testing"""
    with patch('ai_generator.anthropic.Anthropic') as mock_client:
        mock_instance = mock_client.return_value
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        mock_instance.messages.create.return_value = mock_response
        yield mock_instance


@pytest.fixture
def ai_generator(mock_anthropic_client):
    """AIGenerator instance with mocked client"""
    with patch.object(AIGenerator, '__init__', lambda x, *args: None):
        generator = AIGenerator.__new__(AIGenerator)
        generator.client = mock_anthropic_client
        generator.model = "claude-sonnet-4-20250514"
        generator.base_params = {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800
        }
        generator.last_used_tools = []
        generator.last_tool_response = None
        generator.last_tool_results = None
        generator.SYSTEM_PROMPT = "Test system prompt"
        return generator


@pytest.fixture
def temp_vector_store(test_config):
    """Temporary vector store for testing"""
    with patch('chromadb.PersistentClient') as mock_client:
        mock_instance = mock_client.return_value
        mock_collection = Mock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        vector_store = VectorStore(test_config)
        yield vector_store


@pytest.fixture
def session_manager():
    """Session manager for testing"""
    return SessionManager(max_history=2)


@pytest.fixture
def tool_manager(temp_vector_store):
    """Tool manager with mocked dependencies"""
    return ToolManager(temp_vector_store)


@pytest.fixture
def rag_system(test_config):
    """RAG system with mocked dependencies"""
    with patch('rag_system.DocumentProcessor'), \
         patch('rag_system.VectorStore') as mock_vector, \
         patch('rag_system.SessionManager') as mock_session, \
         patch('rag_system.AIGenerator') as mock_ai:
        
        # Configure mocks
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_ai_instance = Mock()
        mock_ai.return_value = mock_ai_instance
        
        rag_system = RAGSystem(test_config)
        yield rag_system


@pytest.fixture
def test_app():
    """FastAPI test application without static files"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    
    # Create test app without mounting static files
    app = FastAPI(title="Test RAG System")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Mock RAG system for API endpoints
    mock_rag = Mock()
    mock_rag.query.return_value = ("Test answer", ["source1"])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    
    # Import and define endpoints inline to avoid static file mount issues
    from pydantic import BaseModel
    from typing import List, Optional, Union, Dict, Any
    from fastapi import HTTPException
    
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, Any]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class NewSessionResponse(BaseModel):
        session_id: str

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/new-session", response_model=NewSessionResponse)
    async def create_new_session():
        try:
            session_id = mock_rag.session_manager.create_session()
            return NewSessionResponse(session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Test API Root"}
    
    # Store mock for test access
    app.state.mock_rag = mock_rag
    return app


@pytest.fixture
def client(test_app):
    """Test client for API testing"""
    return TestClient(test_app)


@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        "course_title": "Test Course",
        "instructor": "Test Instructor", 
        "course_link": "https://example.com/course",
        "lessons": [
            {"number": 1, "title": "Introduction", "lesson_link": "https://example.com/lesson1"},
            {"number": 2, "title": "Advanced Topics", "lesson_link": "https://example.com/lesson2"}
        ],
        "content": "This is test course content for lesson 1.\nThis is test course content for lesson 2."
    }


@pytest.fixture
def mock_course_chunks():
    """Mock course chunks for vector store testing"""
    return [
        {
            "course_title": "Test Course",
            "lesson_number": 1,
            "lesson_title": "Introduction", 
            "content": "This is test content for introduction",
            "chunk_id": "chunk_1"
        },
        {
            "course_title": "Test Course", 
            "lesson_number": 2,
            "lesson_title": "Advanced Topics",
            "content": "This is test content for advanced topics",
            "chunk_id": "chunk_2"
        }
    ]


@pytest.fixture
def temp_docs_dir():
    """Temporary directory with test documents"""
    temp_dir = tempfile.mkdtemp()
    
    # Create test course document
    test_doc = Path(temp_dir) / "test_course.txt"
    test_doc.write_text("""Course Title: Test Course
Course Link: https://example.com/course
Course Instructor: Test Instructor

Lesson 1: Introduction
Lesson Link: https://example.com/lesson1
This is the introduction lesson content.

Lesson 2: Advanced Topics  
Lesson Link: https://example.com/lesson2
This is the advanced topics lesson content.
""")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    yield
    # Cleanup if needed
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]