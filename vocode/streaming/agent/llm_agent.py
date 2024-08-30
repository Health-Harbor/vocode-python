import logging
import re
from typing import AsyncGenerator, Generator, Optional, Tuple

from vocode import getenv
from vocode.streaming.agent.base_agent import BaseAgent, RespondAgent
from vocode.streaming.agent.utils import stream_openai_response_async
from vocode.streaming.models.agent import LLMAgentConfig


class LLMAgent(RespondAgent[LLMAgentConfig]):
    SENTENCE_ENDINGS = [".", "!", "?"]

    DEFAULT_PROMPT_TEMPLATE = "{history}\nHuman: {human_input}\nAI:"

    def __init__(
        self,
        agent_config: LLMAgentConfig,
        logger: Optional[logging.Logger] = None,
        sender="AI",
        recipient="Human",
        openai_api_key: Optional[str] = None,
    ):
        super().__init__(agent_config)
        self.prompt_template = (
            f"{agent_config.prompt_preamble}\n\n{self.DEFAULT_PROMPT_TEMPLATE}"
        )
        self.initial_bot_message = (
            agent_config.initial_message.text if agent_config.initial_message else None
        )
        self.logger = logger or logging.getLogger(__name__)
        self.sender = sender
        self.recipient = recipient
        self.memory = (
            [f"AI: {agent_config.initial_message.text}"]
            if agent_config.initial_message
            else []
        )
        openai_api_key = openai_api_key or getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
        self.stop_tokens = [f"{recipient}:"]
        self.is_first_response = True

    def create_prompt(self, human_input):
        history = "\n".join(self.memory[-5:])
        return self.prompt_template.format(history=history, human_input=human_input)

    def get_memory_entry(self, human_input, response):
        return f"{self.recipient}: {human_input}\n{self.sender}: {response}"

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[str, bool]:
        raise NotImplementedError()

    async def _agen_from_list(self, l):
        for item in l:
            yield item

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError()

    def update_last_bot_message_on_cut_off(self, message: str):
        last_message = self.memory[-1]
        new_last_message = (
            last_message.split("\n", 1)[0] + f"\n{self.sender}: {message}"
        )
        self.memory[-1] = new_last_message
