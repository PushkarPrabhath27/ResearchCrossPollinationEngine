"""
Base Research Agent

Foundation class for all specialized agents using LangChain.
Provides common functionality, LLM setup, memory, and error handling.
"""

import time
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.tools import Tool

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.helpers import save_json

logger = get_logger(__name__)


class BaseResearchAgent(ABC):
    """
    Base class for all research agents
    
    Provides common functionality: LLM setup, memory management,
    tool handling, error recovery, and reasoning tracking.
    """
    
    def __init__(
        self,
        config: Settings,
        tools: Optional[List[Tool]] = None,
        name: str = "BaseAgent",
        temperature: float = 0.7
    ):
        """
        Initialize base agent
        
        Args:
            config: Application configuration
            tools: List of LangChain tools
            name: Agent name
            temperature: LLM temperature
        """
        self.config = config
        self.name = name
        self.temperature = temperature
        self.tools = tools or []
        
        # Setup components
        logger.info(f"Initializing agent: {name}")
        self.llm = self.setup_llm()
        self.memory = self.setup_memory()
        self.agent = self.setup_agent()
        
        # Track reasoning
        self.reasoning_history = []
        self.session_start = time.time()
        
        logger.info(f"Agent {name} initialized successfully")
    
    def setup_llm(self):
        """
        Configure language model based on config
        
        Returns:
            Configured LLM instance
        """
        logger.info(f"Setting up LLM: {self.config.agent.llm_provider}")
        
        try:
            if self.config.agent.llm_provider == "openai":
                return ChatOpenAI(
                    model=self.config.agent.llm_model,
                    temperature=self.temperature,
                    max_tokens=self.config.agent.max_tokens,
                    api_key=self.config.agent.openai_api_key,
                    request_timeout=120
                )
            
            elif self.config.agent.llm_provider == "ollama":
                return Ollama(
                    model=self.config.agent.llm_model,
                    temperature=self.temperature
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.agent.llm_provider}")
                
        except Exception as e:
            logger.error(f"Failed to setup LLM: {e}")
            raise
    
    def setup_memory(self) -> ConversationBufferMemory:
        """
        Configure conversation memory
        
        Returns:
            Memory instance
        """
        logger.debug("Setting up conversation memory")
        
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
    
    def setup_agent(self) -> AgentExecutor:
        """
        Create LangChain agent with tools
        
        Returns:
            Agent executor
        """
        logger.debug(f"Setting up agent with {len(self.tools)} tools")
        
        try:
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=self.memory,
                handle_parsing_errors=True,
                max_iterations=self.config.agent.max_iterations,
                early_stopping_method="generate"
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to setup agent: {e}")
            raise
    
    def run(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute agent with comprehensive error handling
        
        Args:
            query: User's research question
            context: Additional context (field, previous results, etc.)
        
        Returns:
            Dictionary with results, reasoning steps, and metadata
        """
        start_time = time.time()
        
        logger.info(f"Agent {self.name} executing query")
        logger.debug(f"Query: {query[:100]}...")
        
        try:
            # Prepare input
            input_data = {
                "input": query,
                "context": context or {}
            }
            
            # Run agent
            result = self.agent.invoke(input_data)
            
            # Extract reasoning steps
            steps = self._extract_reasoning_steps(result)
            self.reasoning_history.extend(steps)
            
            # Build response
            response = {
                "success": True,
                "output": result.get("output", ""),
                "reasoning_steps": steps,
                "tools_used": self._get_tools_used(result),
                "execution_time": time.time() - start_time,
                "agent_name": self.name
            }
            
            # Log action
            self._log_agent_action("run", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Agent {self.name} failed: {str(e)}", exc_info=True)
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": time.time() - start_time,
                "agent_name": self.name
            }
    
    async def run_async(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute agent asynchronously
        
        Args:
            query: Research question
            context: Additional context
        
        Returns:
            Results dictionary
        """
        start_time = time.time()
        
        logger.info(f"Agent {self.name} executing async query")
        
        try:
            input_data = {
                "input": query,
                "context": context or {}
            }
            
            # Async execution
            result = await self.agent.ainvoke(input_data)
            
            steps = self._extract_reasoning_steps(result)
            self.reasoning_history.extend(steps)
            
            response = {
                "success": True,
                "output": result.get("output", ""),
                "reasoning_steps": steps,
                "tools_used": self._get_tools_used(result),
                "execution_time": time.time() - start_time,
                "agent_name": self.name
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Async execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "agent_name": self.name
            }
    
    def add_tool(self, tool: Tool):
        """
        Add new tool to agent
        
        Args:
            tool: LangChain tool to add
        """
        logger.info(f"Adding tool: {tool.name}")
        
        self.tools.append(tool)
        
        # Reinitialize agent with new tools
        self.agent = self.setup_agent()
    
    def get_reasoning_steps(self) -> List[Dict]:
        """
        Get agent's thought process history
        
        Returns:
            List of reasoning steps
        """
        return self.reasoning_history.copy()
    
    def reset_memory(self):
        """Clear conversation history"""
        logger.info(f"Resetting memory for agent {self.name}")
        
        self.memory.clear()
        self.reasoning_history = []
        self.session_start = time.time()
    
    def export_session(self, output_path: Path):
        """
        Save session for analysis
        
        Args:
            output_path: Path to save session data
        """
        logger.info(f"Exporting session to {output_path}")
        
        session_data = {
            "agent_name": self.name,
            "session_start": self.session_start,
            "session_duration": time.time() - self.session_start,
            "reasoning_history": self.reasoning_history,
            "memory_messages": [
                {"role": m.type, "content": str(m.content)}
                for m in self.memory.chat_memory.messages
            ],
            "total_steps": len(self.reasoning_history)
        }
        
        save_json(session_data, str(output_path))
        logger.info(f"Session exported successfully")
    
    def _extract_reasoning_steps(self, result: Dict) -> List[Dict]:
        """
        Extract agent's thought process from result
        
        Args:
            result: Agent execution result
        
        Returns:
            List of reasoning steps with actions and observations
        """
        steps = []
        
        if "intermediate_steps" in result:
            for i, (action, observation) in enumerate(result["intermediate_steps"]):
                # Truncate long observations
                obs = str(observation)
                if len(obs) > 500:
                    obs = obs[:500] + "... (truncated)"
                
                step_dict = {
                    "step": i + 1,
                    "action": action.tool if hasattr(action, 'tool') else str(action),
                    "action_input": action.tool_input if hasattr(action, 'tool_input') else None,
                    "observation": obs,
                    "timestamp": time.time()
                }
                
                steps.append(step_dict)
        
        return steps
    
    def _get_tools_used(self, result: Dict) -> List[str]:
        """
        Extract list of tools used in execution
        
        Args:
            result: Agent result
        
        Returns:
            List of tool names
        """
        tools_used = set()
        
        if "intermediate_steps" in result:
            for action, _ in result["intermediate_steps"]:
                if hasattr(action, 'tool'):
                    tools_used.add(action.tool)
        
        return list(tools_used)
    
    def _log_agent_action(self, action: str, result: Dict):
        """
        Log agent actions for debugging and analysis
        
        Args:
            action: Action type
            result: Execution result
        """
        logger.info(f"""
Agent Action Log:
  Agent: {self.name}
  Action: {action}
  Success: {result.get('success', False)}
  Time: {result.get('execution_time', 0):.2f}s
  Tools Used: {result.get('tools_used', [])}
  Steps: {len(result.get('reasoning_steps', []))}
        """)
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get agent's system prompt
        
        Must be implemented by subclasses
        
        Returns:
            System prompt string
        """
        pass


# Example usage
if __name__ == "__main__":
    from src.config import get_settings
    from src.utils.logger import setup_logging
    from langchain.tools import Tool
    
    setup_logging(level="INFO")
    config = get_settings()
    
    # Create a simple example agent
    class ExampleAgent(BaseResearchAgent):
        def get_system_prompt(self) -> str:
            return "You are a helpful research assistant."
    
    # Simple tool
    def search_papers(query: str) -> str:
        return f"Found 5 papers for: {query}"
    
    search_tool = Tool(
        name="search_papers",
        func=search_papers,
        description="Search for scientific papers"
    )
    
    # Initialize agent
    print("\n=== Initializing Agent ===")
    agent = ExampleAgent(
        config=config,
        tools=[search_tool],
        name="TestAgent"
    )
    
    # Run query
    print("\n=== Running Query ===")
    result = agent.run(
        query="Find papers about machine learning in cancer research",
        context={"field": "biology", "year": 2024}
    )
    
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Output: {result['output'][:200]}...")
        print(f"Tools used: {result['tools_used']}")
        print(f"Reasoning steps: {len(result['reasoning_steps'])}")
    else:
        print(f"Error: {result.get('error')}")
    
    # Get reasoning history
    print("\n=== Reasoning Steps ===")
    steps = agent.get_reasoning_steps()
    for step in steps:
        print(f"Step {step['step']}: {step['action']}")
    
    # Export session
    print("\n=== Exporting Session ===")
    agent.export_session(Path("./test_session.json"))
    
    print("\n✅ All examples completed!")
