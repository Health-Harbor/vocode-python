from typing import Any, ClassVar, List, Tuple
import json

from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_serializer, model_validator


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(
        extra="ignore",
        protected_namespaces=(),
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _parse_nested_typed_models(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        parsed = dict(data)
        for key, value in data.items():
            if isinstance(value, dict) and "type" in value:
                parsed[key] = TypedModel.model_validate(value)
        return parsed

    def model_dump(self, *args, **kwargs):
        # Preserve subclass fields when nested under a base TypedModel annotation.
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump_json(*args, **kwargs)


class TypedModel(BaseModel):
    _subtypes_: ClassVar[List[Tuple[Any, Any]]] = []

    def __init_subclass__(cls, type=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subtypes_.append((type, cls))

    @classmethod
    def get_cls(_cls, type):
        for t, cls in _cls._subtypes_:
            if t == type:
                return cls
        raise ValueError(f"Unknown type {type}")

    @classmethod
    def get_type(_cls, cls_name):
        for t, cls in _cls._subtypes_:
            if cls.__name__ == cls_name:
                return t
        raise ValueError(f"Unknown class {cls_name}")

    @model_validator(mode="wrap")
    @classmethod
    def _resolve_subtype(cls, data, handler):
        if isinstance(data, dict):
            data_type = data.get("type")
            if data_type is not None:
                sub = cls.get_cls(data_type)
                if sub is not cls:
                    return sub.model_validate(data)
        return handler(data)

    @model_serializer(mode="wrap")
    def _serialize_type(self, serializer):
        data = serializer(self)
        data["type"] = self.get_type(self.__class__.__name__)
        return data

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        if isinstance(json_data, (bytes, bytearray)):
            json_data = json_data.decode()
        return cls.model_validate(json.loads(json_data), **kwargs)

    @property
    def type(self):
        return self.get_type(self.__class__.__name__)
