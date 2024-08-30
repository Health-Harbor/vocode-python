from typing import List, Optional

from pydantic import BaseModel

from vocode import getenv

TEMPLATE = """
Read the following conversation classify the final emotion of the Bot as one of [{emotions}].
Output the degree of emotion as a value between 0 and 1 in the format EMOTION,DEGREE: ex. {example_emotion},0.5

<start>
{{transcript}}
<end>
"""


class BotSentiment(BaseModel):
    emotion: Optional[str] = None
    degree: float = 0.0


class BotSentimentAnalyser:
    def __init__(
        self,
        emotions: List[str],
        model_name: str = "text-davinci-003",
        openai_api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        openai_api_key = openai_api_key or getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
        assert len(emotions) > 0
        self.emotions = [e.lower() for e in emotions]

    async def analyse(self, transcript: str) -> BotSentiment:
        raise NotImplementedError
