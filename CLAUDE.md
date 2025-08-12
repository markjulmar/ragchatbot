# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Create environment file (required)
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

The application serves both the API (`/api/*`) and frontend (`/`) from port 8000.

### Working with Course Documents
Course documents in `docs/` follow this structure:
```
Course Title: [Course Name]
Course Link: [URL]
Course Instructor: [Name]

Lesson 1: [Title]
Lesson Link: [URL]
[Content...]

Lesson 2: [Title]
[Content...]
```

The system automatically processes documents on startup and avoids reprocessing existing courses.

## Dependency Management

### Package Installation Guidelines
- Always use uv to run the server, do not use pip directly.
- Use uv to run Python files too
- **Make sure to use uv to manage all dependencies**

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) system** with a **tool-based AI architecture**. The key architectural pattern is that Claude uses search tools to find relevant course content rather than receiving pre-retrieved context.

### Core Flow Pattern
1. **User Query** → FastAPI endpoint (`/api/query`)
2. **RAG System** orchestrates the pipeline
3. **Claude decides** whether to use search tools based on query type
4. **If course-specific**: Tool executes semantic search via ChromaDB
5. **Claude synthesizes** final response from search results
6. **Session management** maintains conversation context

### Key Components

**Backend Architecture (`backend/`)**:
- `app.py` - FastAPI server with CORS, static file serving, and API endpoints
- `rag_system.py` - Main orchestrator coordinating all components
- `ai_generator.py` - Claude API integration with tool support and conversation handling
- `search_tools.py` - Tool interface for Claude with course search capabilities
- `vector_store.py` - ChromaDB wrapper with semantic search and filtering
- `document_processor.py` - Processes course documents into chunked, searchable format
- `session_manager.py` - Conversation history management with configurable limits
- `models.py` - Pydantic models for Course, Lesson, and CourseChunk data structures
- `config.py` - Centralized configuration with environment variable loading

**Data Models**:
- `Course`: Title (unique ID), instructor, lessons list, course link
- `Lesson`: Number, title, optional lesson link
- `CourseChunk`: Content with course/lesson metadata for vector storage

**Frontend (`frontend/`)**:
- Single-page application with chat interface
- Real-time loading states and error handling
- Markdown rendering for AI responses
- Collapsible sources display from search results

### ChromaDB Collections
- `course_catalog` - Course metadata for semantic course name matching
- `course_content` - Chunked course content with lesson context

### Configuration (`backend/config.py`)
Key settings that affect system behavior:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks
- `MAX_RESULTS: 5` - Maximum search results returned
- `MAX_HISTORY: 2` - Conversation messages remembered
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"`

### Tool System Design
The AI uses a tool-based approach rather than traditional RAG context injection:
- `CourseSearchTool` provides semantic search with course name resolution
- Claude decides autonomously when to search vs. answer from knowledge
- Tool manager handles execution and source tracking
- Sources are extracted and displayed in the frontend

### Session Management
- Sessions auto-created on first query
- History maintained within `MAX_HISTORY` limit (user-assistant pairs)
- Session context passed to Claude for conversation continuity

## Working with the Codebase

### Adding New Course Content
Place documents in `docs/` following the expected format. The system will auto-detect and process new courses on startup without reprocessing existing ones.

### Vector Store Operations
The `VectorStore` class provides unified search with optional course name and lesson number filtering. Course names are semantically matched, so partial names work (e.g., "MCP" matches "MCP: Build Rich-Context AI Apps").

### Tool Development
New tools should implement the `Tool` interface from `search_tools.py` with `get_tool_definition()` and `execute()` methods. Register tools with `ToolManager` to make them available to Claude.

### Error Handling Patterns
- Vector store operations return `SearchResults` objects with optional error messages
- API endpoints use HTTPException for client errors
- Frontend displays loading states and graceful error recovery
```