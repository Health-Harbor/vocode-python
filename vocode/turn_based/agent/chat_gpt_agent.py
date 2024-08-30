from typing import Optional

import openai

from vocode import getenv
from vocode.turn_based.agent.base_agent import BaseAgent


class ChatGPTAgent(BaseAgent):
    def __init__(
        self,
        system_prompt: str,
        api_key: Optional[str] = None,
        initial_message: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 100,
    ):
        super().__init__(initial_message=initial_message)
        openai.api_key = getenv("OPENAI_API_KEY", api_key)
        if not openai.api_key:
            raise ValueError("OpenAI API key not provided")

    def respond(self, human_input: str):
        raise NotImplementedError
