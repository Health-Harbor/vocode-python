import logging
from typing import AsyncGenerator, Optional, Tuple

from vocode import getenv
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.agent.utils import get_sentence_from_buffer
from vocode.streaming.models.agent import ChatAnthropicAgentConfig

SENTENCE_ENDINGS = [".", "!", "?"]


class ChatAnthropicAgent(RespondAgent[ChatAnthropicAgentConfig]):
    def __init__(
        self,
        agent_config: ChatAnthropicAgentConfig,
        logger: Optional[logging.Logger] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        super().__init__(agent_config=agent_config, logger=logger)

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[str, bool]:
        raise NotImplementedError()

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError()

    def update_last_bot_message_on_cut_off(self, message: str):
        raise NotImplementedError()
