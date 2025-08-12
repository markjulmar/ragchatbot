from unittest.mock import MagicMock, Mock, patch

import pytest
from ai_generator import AIGenerator
from rag_system import RAGSystem, ToolRound
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import VectorStore


class TestSequentialToolCalling:
    """Test suite for sequential tool calling functionality"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration object"""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "test_path"
        config.EMBEDDING_MODEL = "test_model"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test_key"
        config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
        config.MAX_HISTORY = 2
        return config

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_client:
            yield mock_client.return_value

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator instance with mocked client"""
        return AIGenerator("test_key", "claude-3-sonnet-20240229")

    @pytest.fixture
    def rag_system(self, mock_config):
        """Create RAGSystem instance with mocked dependencies"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.SessionManager"),
        ):
            return RAGSystem(mock_config)

    def test_single_round_sufficient_no_additional_tools(
        self, ai_generator, mock_anthropic_client
    ):
        """Test case where single round is sufficient - no additional tools needed"""
        # Mock response without tool use
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="This is a direct answer without tools")]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Test
        result = ai_generator.generate_response_with_tools(
            messages=[{"role": "user", "content": "What is Python?"}],
            tools=[],
            tool_manager=Mock(),
        )

        # Verify
        assert result == "This is a direct answer without tools"
        assert ai_generator.last_used_tools == []
        assert mock_anthropic_client.messages.create.call_count == 1

    def test_two_rounds_needed_sequential_tool_usage(
        self, ai_generator, mock_anthropic_client
    ):
        """Test case where two rounds are needed for sequential tool usage"""
        # Mock first round - tool usage
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_course_outline"
        mock_tool_block.input = {"course_title": "MCP"}
        mock_tool_block.id = "tool_1"
        mock_tool_response.content = [mock_tool_block]

        mock_anthropic_client.messages.create.return_value = mock_tool_response

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Course outline results"

        # Test
        result = ai_generator.generate_response_with_tools(
            messages=[{"role": "user", "content": "Get outline for MCP course"}],
            tools=[{"name": "get_course_outline"}],
            tool_manager=mock_tool_manager,
        )

        # Verify
        assert result == "Tool execution completed - continuing to next round"
        assert len(ai_generator.last_used_tools) == 1
        assert ai_generator.last_used_tools[0]["name"] == "get_course_outline"
        assert ai_generator.last_used_tools[0]["input"] == {"course_title": "MCP"}
        assert ai_generator.last_tool_response == [mock_tool_block]
        assert len(ai_generator.last_tool_results) == 1
        mock_tool_manager.execute_tool.assert_called_once_with(
            "get_course_outline", course_title="MCP"
        )

    def test_loop_prevention_same_tool_same_params(self, rag_system):
        """Test loop prevention when same tool is called with same parameters"""
        # Mock tool calls that would create a loop
        current_calls = [{"name": "search_course_content", "input": {"query": "test"}}]
        previous_calls = [{"name": "search_course_content", "input": {"query": "test"}}]

        # Test
        is_loop = rag_system._detect_tool_loop(current_calls, previous_calls)

        # Verify
        assert is_loop == True

    def test_no_loop_different_tool_params(self, rag_system):
        """Test no loop when tools have different parameters"""
        # Mock tool calls with different parameters
        current_calls = [{"name": "search_course_content", "input": {"query": "test1"}}]
        previous_calls = [
            {"name": "search_course_content", "input": {"query": "test2"}}
        ]

        # Test
        is_loop = rag_system._detect_tool_loop(current_calls, previous_calls)

        # Verify
        assert is_loop == False

    def test_final_response_synthesis_without_tools(
        self, ai_generator, mock_anthropic_client
    ):
        """Test final response generation without tools for synthesis"""
        # Mock final response
        mock_response = Mock()
        mock_response.content = [Mock(text="Final synthesized answer")]
        mock_anthropic_client.messages.create.return_value = mock_response

        # Test
        messages = [
            {"role": "user", "content": "Query"},
            {"role": "assistant", "content": "Tool response"},
            {"role": "user", "content": "Tool results"},
        ]
        result = ai_generator.generate_final_response(messages=messages)

        # Verify
        assert result == "Final synthesized answer"
        # Verify no tools were passed in the final call
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tools" not in call_args[1]

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.SessionManager")
    def test_rag_system_sequential_execution_flow(
        self, mock_session, mock_vector, mock_doc, mock_config
    ):
        """Test complete RAGSystem sequential execution flow"""
        # Setup
        rag_system = RAGSystem(mock_config)

        # Mock AI generator responses
        rag_system.ai_generator.generate_response_with_tools = Mock()
        rag_system.ai_generator.generate_final_response = Mock()
        rag_system.ai_generator.last_used_tools = [{"name": "test_tool", "input": {}}]
        rag_system.ai_generator.last_tool_response = ["tool response"]
        rag_system.ai_generator.last_tool_results = ["tool results"]

        # Mock tool manager
        rag_system.tool_manager.get_last_sources = Mock(return_value=["source1"])
        rag_system.tool_manager.reset_sources = Mock()

        # First call returns tool usage, second call no tools
        rag_system.ai_generator.generate_response_with_tools.side_effect = [
            "Round 1 response",
            "Round 2 response",
        ]

        # Second call indicates no tools used
        def side_effect_last_tools(*args, **kwargs):
            if rag_system.ai_generator.generate_response_with_tools.call_count >= 2:
                return []
            return [{"name": "test_tool", "input": {}}]

        # Mock the last_used_tools dynamically
        type(rag_system.ai_generator).last_used_tools = property(
            lambda x: side_effect_last_tools()
        )
        rag_system.ai_generator.generate_final_response.return_value = "Final response"

        # Test
        response, sources = rag_system._execute_sequential_rounds("test query", None)

        # Verify
        assert response == "Final response"  # Final response generated after max rounds
        assert sources == ["source1", "source1"]  # Sources from both rounds

    def test_tool_execution_error_handling(self, ai_generator, mock_anthropic_client):
        """Test error handling during tool execution"""
        # Mock response with tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "failing_tool"
        mock_tool_block.input = {"param": "value"}
        mock_tool_block.id = "tool_1"
        mock_tool_response.content = [mock_tool_block]

        mock_anthropic_client.messages.create.return_value = mock_tool_response

        # Mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        # Test - should raise exception since current implementation doesn't handle errors
        with pytest.raises(Exception, match="Tool execution failed"):
            result = ai_generator.generate_response_with_tools(
                messages=[{"role": "user", "content": "Test"}],
                tools=[{"name": "failing_tool"}],
                tool_manager=mock_tool_manager,
            )
    def test_max_rounds_enforcement(self, rag_system):
        """Test that maximum rounds (2) is enforced"""
        # Setup mocks to simulate continuous tool usage
        rag_system.ai_generator.generate_response_with_tools = Mock(
            return_value="Round response"
        )
        rag_system.ai_generator.generate_final_response = Mock(
            return_value="Final response"
        )
        rag_system.ai_generator.last_used_tools = [
            {"name": "persistent_tool", "input": {}}
        ]
        rag_system.ai_generator.last_tool_response = ["response"]
        rag_system.ai_generator.last_tool_results = ["results"]
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()

        # Test
        response, sources = rag_system._execute_sequential_rounds("test query", None)

        # Verify - should call generate_response_with_tools exactly 2 times (max rounds)
        assert rag_system.ai_generator.generate_response_with_tools.call_count == 2
        # Should call final response generation once
        assert rag_system.ai_generator.generate_final_response.call_count == 1
        assert response == "Final response"

    def test_conversation_history_preservation(
        self, ai_generator, mock_anthropic_client
    ):
        """Test that conversation history is properly preserved between rounds"""
        # Mock response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response with history")]
        mock_anthropic_client.messages.create.return_value = mock_response

        history = "Previous conversation context"

        # Test
        result = ai_generator.generate_response_with_tools(
            messages=[{"role": "user", "content": "Test query"}],
            conversation_history=history,
        )

        # Verify
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert history in system_content
        assert ai_generator.SYSTEM_PROMPT in system_content

    def test_source_accumulation_across_rounds(self, rag_system):
        """Test that sources are properly accumulated across multiple rounds"""
        # Setup mocks
        rag_system.ai_generator.generate_response_with_tools = Mock()
        rag_system.ai_generator.generate_final_response = Mock(return_value="Final")
        rag_system.ai_generator.last_tool_response = ["response"]
        rag_system.ai_generator.last_tool_results = ["results"]

        # Mock different sources for each round
        source_sequence = [["source1", "source2"], ["source3"], []]
        rag_system.tool_manager.get_last_sources = Mock(side_effect=source_sequence)
        rag_system.tool_manager.reset_sources = Mock()

        # Simulate tool usage in first round, none in second
        calls_made = []
        def mock_tools_side_effect(*args, **kwargs):
            calls_made.append(True)
            if len(calls_made) == 1:
                # First round - tools used
                rag_system.ai_generator.last_used_tools = [{"name": "tool1"}]
                return "Round 1"
            else:
                # Second round - no tools used, terminate naturally
                rag_system.ai_generator.last_used_tools = []
                return "Round 2"

        rag_system.ai_generator.generate_response_with_tools.side_effect = (
            mock_tools_side_effect
        )

        # Test
        response, sources = rag_system._execute_sequential_rounds("test query", None)

        # Verify source accumulation
        expected_sources = ["source1", "source2", "source3"]
        assert sources == expected_sources


if __name__ == "__main__":
    pytest.main([__file__])
