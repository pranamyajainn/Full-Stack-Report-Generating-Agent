from typing import Any, Dict, List, Optional

from llama_index.core.agent.multi_agent.agent_config import AgentConfig
from llama_index.core.agent.multi_agent.workflow_events import HandoffEvent, ToolCall, LLMEvent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.tools import BaseTool, AsyncBaseTool, adapt_to_async_tool
from llama_index.core.workflow import Context, FunctionToolWithContext, StartEvent, StopEvent, Workflow, step
from llama_index.core.settings import Settings


async def handoff(ctx: Context, to_agent: str, reason: str) -> HandoffEvent:
    """Handoff to the given agent."""
    ctx.send_event(
        HandoffEvent(
            from_agent=await ctx.get("current_agent"), 
            to_agent=to_agent, 
            reason=reason
        )
    )

class MultiAgentSystem(Workflow):
    """A workflow for managing multiple agents with handoffs."""
    
    def __init__(
        self,
        agent_configs: List[AgentConfig],
        initial_state: Optional[Dict] = None,
        memory: Optional[BaseMemory] = None,
        timeout: Optional[float] = None,
        **workflow_kwargs: Any
    ):
        super().__init__(timeout=timeout, **workflow_kwargs)
        if not agent_configs:
            raise ValueError("At least one agent config must be provided")
        
        self.agent_configs = {cfg.name: cfg for cfg in agent_configs}
        only_one_root_agent = sum(cfg.is_root_agent for cfg in agent_configs) == 1
        if not only_one_root_agent:
            raise ValueError("Exactly one root agent must be provided")
        
        self.root_agent = next(cfg.name for cfg in agent_configs if cfg.is_root_agent)

        self.initial_state = initial_state or {}
        self.memory = memory or ChatMemoryBuffer.from_defaults(
            llm=Settings.llm or agent_configs[0].llm
        )

    def _create_handoff_tool(self, agent_config: AgentConfig) -> BaseTool:
        """Creates a handoff tool for the given agent."""
        # TODO: Implement handoff tool creation
        pass

    async def _init_context(self, ctx: Context) -> None:
        """Initialize the context once, if needed."""
        if not await ctx.get("memory"):
            await ctx.set("memory", self.memory)
        if not await ctx.get("agent_configs"):
            await ctx.set("agent_configs", self.agent_configs)
        if not await ctx.get("current_state"):
            await ctx.set("current_state", self.initial_state)
        if not await ctx.get("current_agent"):
            await ctx.set("current_agent", self.root_agent)
            
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> HandoffEvent:
        """Sets up the workflow and validates inputs."""
        user_msg = ev.get("user_msg")
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role="user", content=user_msg)
            
        # Store everything in context
        await ctx.set("memory", self.memory)
        await ctx.set("agent_configs", self.agent_configs)
        await ctx.set("current_state", self.initial_state)
        
        # Add message to memory
        self.memory.put(user_msg)
        
        # Start with first agent
        first_agent = list(self.agent_configs.keys())[0]
        return HandoffEvent(
            from_agent="system",
            to_agent=first_agent,
            reason="Initial message"
        )

    @step
    async def handle_agent(
        self, ctx: Context, ev: HandoffEvent
    ) -> HandoffEvent | StopEvent:
        """Main agent handling logic."""
        agent_config = self.agent_configs[ev.to_agent]
        memory = await ctx.get("memory")
        current_state = await ctx.get("current_state")
        
        # Create tools including handoff capability
        tools = list(agent_config.tools or [])
        if agent_config.tool_retriever:
            tools.extend(agent_config.tool_retriever.retrieve())
        handoff_tool = self._create_handoff_tool(agent_config)
        tools.append(handoff_tool)
        
        # TODO: Implement agent interaction logic
        # - Handle tool calls
        # - Stream LLM events
        # - Handle handoffs
        # - Return appropriate events
        
        pass

    async def run(
        self,
        user_msg: str | ChatMessage,
        **kwargs: Any
    ):
        """Run the multi-agent workflow."""
        return await super().run(user_msg=user_msg, **kwargs)
     