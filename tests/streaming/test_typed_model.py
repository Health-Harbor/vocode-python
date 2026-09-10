import pytest
from pydantic import ValidationError

from vocode.streaming.models.agent import (
    AgentConfig,
    ChatGPTAgentConfig,
    FillerAudioConfig,
)
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import (
    AzureSynthesizerConfig,
    ElevenLabsSynthesizerConfig,
)
from vocode.streaming.models.telephony import TwilioCallConfig, TwilioConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
    TranscriberConfig,
)
from vocode.streaming.models.websocket import AudioMessage, ReadyMessage, WebSocketMessage


def test_typed_model_constructor_exposes_type():
    config = ChatGPTAgentConfig(prompt_preamble="hello")
    assert config.type == "agent_chat_gpt"
    assert isinstance(config, ChatGPTAgentConfig)


def test_typed_model_validate_dispatches_subclass():
    config = AgentConfig.model_validate(
        {
            "type": "agent_chat_gpt",
            "prompt_preamble": "hello",
            "initial_message": {"type": "message_base", "text": "hi"},
        }
    )
    assert isinstance(config, ChatGPTAgentConfig)
    assert config.prompt_preamble == "hello"
    assert isinstance(config.initial_message, BaseMessage)
    assert config.initial_message.text == "hi"


def test_typed_model_dump_includes_type():
    config = ChatGPTAgentConfig(
        prompt_preamble="hello",
        initial_message=BaseMessage(text="hi"),
    )
    dumped = config.model_dump()
    assert dumped["type"] == "agent_chat_gpt"
    assert dumped["initial_message"]["type"] == "message_base"
    assert dumped["initial_message"]["text"] == "hi"


def test_typed_model_json_roundtrip():
    original = ChatGPTAgentConfig(prompt_preamble="hello")
    restored = AgentConfig.model_validate_json(original.model_dump_json())
    assert isinstance(restored, ChatGPTAgentConfig)
    assert restored.prompt_preamble == "hello"


def test_nested_endpointing_config_roundtrip():
    transcriber = DeepgramTranscriberConfig(
        sampling_rate=8000,
        audio_encoding=AudioEncoding.MULAW,
        chunk_size=320,
        endpointing_config=PunctuationEndpointingConfig(),
    )
    restored = TranscriberConfig.model_validate(transcriber.model_dump())
    assert isinstance(restored, DeepgramTranscriberConfig)
    assert isinstance(restored.endpointing_config, PunctuationEndpointingConfig)


def test_call_config_json_roundtrip():
    call_config = TwilioCallConfig(
        transcriber_config=TwilioCallConfig.default_transcriber_config(),
        agent_config=ChatGPTAgentConfig(prompt_preamble="hello"),
        synthesizer_config=TwilioCallConfig.default_synthesizer_config(),
        twilio_config=TwilioConfig(account_sid="sid", auth_token="token"),
        twilio_sid="CA123",
        from_phone="+15555550100",
        to_phone="+15555550101",
    )
    restored = TwilioCallConfig.model_validate_json(call_config.model_dump_json())
    assert restored.twilio_sid == "CA123"
    assert isinstance(restored.agent_config, ChatGPTAgentConfig)
    assert isinstance(restored.synthesizer_config, AzureSynthesizerConfig)
    assert isinstance(restored.transcriber_config, DeepgramTranscriberConfig)


def test_websocket_message_dispatch():
    ready = ReadyMessage()
    parsed = WebSocketMessage.model_validate_json(ready.model_dump_json())
    assert isinstance(parsed, ReadyMessage)

    audio = AudioMessage.from_bytes(b"abc")
    parsed_audio = WebSocketMessage.model_validate(audio.model_dump())
    assert isinstance(parsed_audio, AudioMessage)
    assert parsed_audio.get_bytes() == b"abc"


def test_filler_audio_validator_prefers_typing_noise():
    config = FillerAudioConfig(use_phrases=True, use_typing_noise=True)
    assert config.use_typing_noise is True
    assert config.use_phrases is False


def test_filler_audio_validator_requires_one_source():
    with pytest.raises(ValidationError):
        FillerAudioConfig(use_phrases=False, use_typing_noise=False)


def test_eleven_labs_stability_and_similarity_must_be_paired():
    with pytest.raises(ValidationError):
        ElevenLabsSynthesizerConfig(
            sampling_rate=16000,
            audio_encoding=AudioEncoding.LINEAR16,
            stability=0.5,
        )
