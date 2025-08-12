import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **Course Content Search** - Use for questions about specific course content, lessons, or detailed educational materials
2. **Course Outline** - Use for questions about course structure, lesson lists, or complete course outlines

Sequential Tool Usage Guidelines:
- **Multi-step queries**: You can use tools in sequence to answer complex questions requiring multiple pieces of information
- **Example workflow**: First get course outline to understand structure → then search for specific content within that course
- **Maximum 2 tool rounds**: You have up to 2 opportunities to use tools before providing your final answer
- **Reasoning between rounds**: Use results from previous tool calls to inform subsequent tool usage
- **Complex comparisons**: Use multiple searches to gather information for comparative analysis

Tool Usage Strategy:
- **Course outline queries**: Use the course outline tool to get complete course information including title, course link, and all lessons with their numbers and titles
- **Content-specific questions**: Use the search tool for questions about specific educational materials or lesson content
- **Multi-part questions**: Break down into sequential tool calls as needed
- **Synthesize all results**: Combine information from all tool calls into comprehensive answers
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use the course outline tool to get complete course structure, lesson lists, and course information
- **Course content questions**: Use the search tool for specific educational materials or lesson content
- **Complex queries**: Use tools sequentially to gather all necessary information before answering
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        
        # Track tool usage for sequential calling
        self.last_used_tools = []
        self.last_tool_response = None
        self.last_tool_results = None
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def generate_response_with_tools(self, messages: List[Dict], 
                                    conversation_history: Optional[str] = None,
                                    tools: Optional[List] = None,
                                    tool_manager=None) -> str:
        """
        Generate AI response with tools enabled for sequential calling.
        
        Args:
            messages: Message chain from previous rounds
            conversation_history: Previous conversation context
            tools: Available tools
            tool_manager: Manager to execute tools
            
        Returns:
            Response after potential tool execution
        """
        # Reset tracking variables
        self.last_used_tools = []
        self.last_tool_response = None
        self.last_tool_results = None
        
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution_sequential(response, api_params, tool_manager)
        
        # No tools used - return direct response
        return response.content[0].text
    
    def _handle_tool_execution_sequential(self, initial_response, base_params: Dict, tool_manager):
        """
        Handle tool execution for sequential calling, tracking results.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Response text after tool execution (for continuation)
        """
        # Track tool usage for loop detection
        self.last_used_tools = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                self.last_used_tools.append({
                    'name': content_block.name,
                    'input': content_block.input
                })
        
        # Store the assistant's response for message chaining
        self.last_tool_response = initial_response.content
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Store tool results for message chaining
        self.last_tool_results = tool_results
        
        # For sequential calling, we return a placeholder that indicates tools were used
        # The actual synthesis will happen in the final round
        return "Tool execution completed - continuing to next round"
    
    def generate_final_response(self, messages: List[Dict], 
                               conversation_history: Optional[str] = None) -> str:
        """
        Generate final response without tools for synthesis.
        
        Args:
            messages: Complete message chain from all rounds
            conversation_history: Previous conversation context
            
        Returns:
            Final synthesized response
        """
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text