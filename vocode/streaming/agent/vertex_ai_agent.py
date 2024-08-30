import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.agent import ChatVertexAIAgentConfig


class ChatVertexAIAgent(RespondAgent[ChatVertexAIAgentConfig]):
    def __init__(
        self,
        agent_config: ChatVertexAIAgentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(agent_config=agent_config, logger=logger)
        if agent_config.initial_message:
            raise NotImplementedError("initial_message not supported for Vertex AI")
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[str, bool]:
        raise NotImplementedError("respond not supported for Vertex AI")
