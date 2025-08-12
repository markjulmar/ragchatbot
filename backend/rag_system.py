import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ai_generator import AIGenerator
from document_processor import DocumentProcessor
from models import Course, CourseChunk, Lesson
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import VectorStore


@dataclass
class ToolRound:
    """Tracks state for a single tool execution round"""

    round_number: int
    tools_used: List[str]
    tool_results: List[Dict]
    accumulated_sources: List[Dict]
    messages: List[Dict]


class RAGSystem:
    """Main orchestrator for the Retrieval-Augmented Generation system"""

    def __init__(self, config):
        self.config = config

        # Initialize core components
        self.document_processor = DocumentProcessor(
            config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        self.ai_generator = AIGenerator(
            config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL
        )
        self.session_manager = SessionManager(config.MAX_HISTORY)

        # Initialize search tools
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.outline_tool = CourseOutlineTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)
        self.tool_manager.register_tool(self.outline_tool)

    def add_course_document(self, file_path: str) -> Tuple[Course, int]:
        """
        Add a single course document to the knowledge base.

        Args:
            file_path: Path to the course document

        Returns:
            Tuple of (Course object, number of chunks created)
        """
        try:
            # Process the document
            course, course_chunks = self.document_processor.process_course_document(
                file_path
            )

            # Add course metadata to vector store for semantic search
            self.vector_store.add_course_metadata(course)

            # Add course content chunks to vector store
            self.vector_store.add_course_content(course_chunks)

            return course, len(course_chunks)
        except Exception as e:
            print(f"Error processing course document {file_path}: {e}")
            return None, 0

    def add_course_folder(
        self, folder_path: str, clear_existing: bool = False
    ) -> Tuple[int, int]:
        """
        Add all course documents from a folder.

        Args:
            folder_path: Path to folder containing course documents
            clear_existing: Whether to clear existing data first

        Returns:
            Tuple of (total courses added, total chunks created)
        """
        total_courses = 0
        total_chunks = 0

        # Clear existing data if requested
        if clear_existing:
            print("Clearing existing data for fresh rebuild...")
            self.vector_store.clear_all_data()

        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return 0, 0

        # Get existing course titles to avoid re-processing
        existing_course_titles = set(self.vector_store.get_existing_course_titles())

        # Process each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(
                (".pdf", ".docx", ".txt")
            ):
                try:
                    # Check if this course might already exist
                    # We'll process the document to get the course ID, but only add if new
                    course, course_chunks = (
                        self.document_processor.process_course_document(file_path)
                    )

                    if course and course.title not in existing_course_titles:
                        # This is a new course - add it to the vector store
                        self.vector_store.add_course_metadata(course)
                        self.vector_store.add_course_content(course_chunks)
                        total_courses += 1
                        total_chunks += len(course_chunks)
                        print(
                            f"Added new course: {course.title} ({len(course_chunks)} chunks)"
                        )
                        existing_course_titles.add(course.title)
                    elif course:
                        print(f"Course already exists: {course.title} - skipping")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        return total_courses, total_chunks

    def query(
        self, query: str, session_id: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Process a user query using the RAG system with sequential tool calling support.

        Args:
            query: User's question
            session_id: Optional session ID for conversation context

        Returns:
            Tuple of (response, sources list from all tool rounds)
        """
        # Create prompt for the AI with clear instructions
        prompt = f"""Answer this question about course materials: {query}"""

        # Get conversation history if session exists
        history = None
        if session_id:
            history = self.session_manager.get_conversation_history(session_id)

        # Execute sequential tool calling with up to 2 rounds
        response, all_sources = self._execute_sequential_rounds(prompt, history)

        # Update conversation history
        if session_id:
            self.session_manager.add_exchange(session_id, query, response)

        return response, all_sources

    def _execute_sequential_rounds(
        self, query: str, history: Optional[str]
    ) -> Tuple[str, List[str]]:
        """
        Execute up to 2 rounds of tool calling, allowing Claude to reason about previous results.

        Args:
            query: The user's question
            history: Conversation history

        Returns:
            Tuple of (final response, accumulated sources)
        """
        max_rounds = 2
        accumulated_sources = []
        used_tool_calls = []  # Track tool calls to prevent loops
        messages = [{"role": "user", "content": query}]

        for round_num in range(1, max_rounds + 1):
            # Reset tool sources for this round
            self.tool_manager.reset_sources()

            # Generate response with tools enabled
            response = self.ai_generator.generate_response_with_tools(
                messages=messages,
                conversation_history=history,
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager,
            )

            # Get sources from this round
            round_sources = self.tool_manager.get_last_sources()
            accumulated_sources.extend(round_sources)

            # Check termination conditions
            if not self.ai_generator.last_used_tools:
                # No tools used - natural termination
                return response, accumulated_sources

            # Check for tool loops
            current_tool_calls = self.ai_generator.last_used_tools
            if self._detect_tool_loop(current_tool_calls, used_tool_calls):
                # Loop detected - synthesize final response without tools
                break

            # Track tool usage
            used_tool_calls.extend(current_tool_calls)

            # Add assistant's response and tool results to message chain
            messages.append(
                {"role": "assistant", "content": self.ai_generator.last_tool_response}
            )
            messages.append(
                {"role": "user", "content": self.ai_generator.last_tool_results}
            )

            # If this was the last allowed round, break
            if round_num >= max_rounds:
                break

        # Generate final response without tools for synthesis
        final_response = self.ai_generator.generate_final_response(
            messages=messages, conversation_history=history
        )

        return final_response, accumulated_sources

    def _detect_tool_loop(
        self, current_calls: List[Dict], previous_calls: List[Dict]
    ) -> bool:
        """
        Detect if the same tool is being called with the same parameters.

        Args:
            current_calls: Tool calls from current round
            previous_calls: All previous tool calls

        Returns:
            True if a loop is detected
        """
        for current_call in current_calls:
            for previous_call in previous_calls:
                if current_call.get("name") == previous_call.get(
                    "name"
                ) and current_call.get("input") == previous_call.get("input"):
                    return True
        return False

    def get_course_analytics(self) -> Dict:
        """Get analytics about the course catalog"""
        return {
            "total_courses": self.vector_store.get_course_count(),
            "course_titles": self.vector_store.get_existing_course_titles(),
        }
