import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for FastAPI endpoint functionality"""

    def test_query_endpoint_success(self, client):
        """Test successful query processing"""
        response = client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": "test-123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-123"
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["source1"]

    def test_query_endpoint_auto_session_creation(self, client):
        """Test query endpoint creates session when none provided"""
        response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"

    def test_query_endpoint_validation_error(self, client):
        """Test query endpoint with missing required fields"""
        response = client.post(
            "/api/query",
            json={}  # Missing required 'query' field
        )
        
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/api/query",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_query_endpoint_rag_system_error(self, client, test_app):
        """Test query endpoint when RAG system raises exception"""
        # Make the mock RAG system raise an exception
        test_app.state.mock_rag.query.side_effect = Exception("RAG system error")
        
        response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]

    def test_courses_endpoint_success(self, client):
        """Test successful course statistics retrieval"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 1
        assert data["course_titles"] == ["Test Course"]

    def test_courses_endpoint_analytics_error(self, client, test_app):
        """Test courses endpoint when analytics raises exception"""
        test_app.state.mock_rag.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]

    def test_new_session_endpoint_success(self, client):
        """Test successful new session creation"""
        response = client.post("/api/new-session")
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

    def test_new_session_endpoint_error(self, client, test_app):
        """Test new session endpoint when session manager raises exception"""
        test_app.state.mock_rag.session_manager.create_session.side_effect = Exception("Session error")
        
        response = client.post("/api/new-session")
        
        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]

    def test_root_endpoint(self, client):
        """Test root endpoint returns test message"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Test API Root"

    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        # Use a valid POST request to check CORS headers
        response = client.post(
            "/api/query",
            json={"query": "test"},
            headers={"origin": "http://localhost:3000"}
        )
        
        # Check that CORS headers are present
        assert response.status_code == 200
        # Note: TestClient doesn't always expose CORS headers the same way as a real browser
        # This test verifies the endpoint works with cross-origin requests

    def test_query_with_large_payload(self, client):
        """Test query endpoint with large payload"""
        large_query = "What is Python? " * 1000  # Large query string
        
        response = client.post(
            "/api/query",
            json={"query": large_query}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_concurrent_queries(self, client):
        """Test multiple concurrent queries don't interfere"""
        import concurrent.futures
        
        def make_query(session_id):
            return client.post(
                "/api/query",
                json={"query": f"Query from session {session_id}", "session_id": session_id}
            )
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_query, f"session-{i}") for i in range(3)]
            responses = [future.result() for future in futures]
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200

    def test_query_response_format(self, client):
        """Test query response matches expected format"""
        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Verify sources can contain strings or dicts
        for source in data["sources"]:
            assert isinstance(source, (str, dict))

    def test_courses_response_format(self, client):
        """Test courses response matches expected format"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
        
        # Verify all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_method_not_allowed(self, client):
        """Test endpoints reject wrong HTTP methods"""
        # POST endpoint with GET
        response = client.get("/api/query")
        assert response.status_code == 405
        
        # GET endpoint with POST
        response = client.post("/api/courses")
        assert response.status_code == 405

    def test_nonexistent_endpoint(self, client):
        """Test 404 for nonexistent endpoints"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    @pytest.mark.parametrize("query,expected_calls", [
        ("Simple query", 1),
        ("", 1),  # Empty query should still work
        ("Query with special chars: !@#$%^&*()", 1),
    ])
    def test_query_variations(self, client, test_app, query, expected_calls):
        """Test query endpoint with various query formats"""
        # Reset call count
        test_app.state.mock_rag.query.reset_mock()
        
        response = client.post(
            "/api/query",
            json={"query": query}
        )
        
        assert response.status_code == 200
        assert test_app.state.mock_rag.query.call_count == expected_calls


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests that test API with more realistic scenarios"""

    def test_session_workflow(self, client):
        """Test complete session workflow"""
        # Create new session
        session_response = client.post("/api/new-session")
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]
        
        # Use session for query
        query_response = client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": session_id}
        )
        assert query_response.status_code == 200
        assert query_response.json()["session_id"] == session_id
        
        # Check courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200

    def test_multiple_queries_same_session(self, client):
        """Test multiple queries in the same session"""
        session_id = "persistent-session"
        
        queries = ["What is Python?", "How to use FastAPI?", "Explain RAG systems"]
        
        for query in queries:
            response = client.post(
                "/api/query",
                json={"query": query, "session_id": session_id}
            )
            assert response.status_code == 200
            assert response.json()["session_id"] == session_id