from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .errors import ToolContractError
from .models import DecodedToolCall


_VLLM_GRAMMAR_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "anyOf",
        "contains",
        "format",
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
        "multipleOf",
        "oneOf",
        "patternProperties",
        "propertyNames",
        "uniqueItems",
    }
)
_SCHEMA_MAP_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "properties",
    }
)


def _vllm_grammar_schema(value: Any, *, schema_map: bool = False) -> Any:
    if isinstance(value, list):
        return [_vllm_grammar_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    if schema_map:
        return {
            key: _vllm_grammar_schema(item)
            for key, item in value.items()
        }
    return {
        key: _vllm_grammar_schema(item, schema_map=key in _SCHEMA_MAP_KEYS)
        for key, item in value.items()
        if key not in _VLLM_GRAMMAR_UNSUPPORTED_SCHEMA_KEYS
    }


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    arguments_type: type
    json_schema: dict[str, Any]
    contextual_schema: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def schema_for(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if context is None or self.contextual_schema is None:
            return self.json_schema
        try:
            schema = self.contextual_schema(context)
        except Exception as error:
            raise ToolContractError(
                f"Tool {self.name} не смог построить контекстную JSON Schema"
            ) from error
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise ToolContractError(
                f"Tool {self.name} построил невалидную контекстную JSON Schema"
            ) from error
        return schema

    def openai_schema(
        self,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _vllm_grammar_schema(self.schema_for(context)),
            },
        }


class TypedToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._validators: dict[str, Draft202012Validator] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ToolContractError(f"Tool {spec.name} уже зарегистрирован")
        try:
            Draft202012Validator.check_schema(spec.json_schema)
        except SchemaError as error:
            raise ToolContractError(f"Tool {spec.name} содержит невалидную JSON Schema") from error
        self._tools[spec.name] = spec
        self._validators[spec.name] = Draft202012Validator(spec.json_schema)

    def openai_schemas(
        self,
        allowed_names: tuple[str, ...],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [self._get(name).openai_schema(context) for name in allowed_names]

    def decode(
        self,
        raw_call: dict[str, Any],
        allowed_names: tuple[str, ...],
        *,
        context: dict[str, Any] | None = None,
    ) -> DecodedToolCall:
        name = str(raw_call.get("name", ""))
        if name not in allowed_names:
            raise ToolContractError(f"Tool {name or '<пусто>'} не разрешён для этого промпта")
        spec = self._get(name)
        arguments = raw_call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ToolContractError("Аргументы tool call содержат невалидный JSON") from error
        if not isinstance(arguments, dict):
            raise ToolContractError("Аргументы tool call должны быть объектом")
        validator = (
            self._validators[name]
            if context is None or spec.contextual_schema is None
            else Draft202012Validator(spec.schema_for(context))
        )
        self._validate_schema(arguments, validator)
        try:
            typed = spec.arguments_type(**arguments)
        except (TypeError, ValueError) as error:
            raise ToolContractError(f"Аргументы {name} не соответствуют типу: {error}") from error
        return DecodedToolCall(
            call_id=str(raw_call.get("id", "")),
            name=name,
            arguments=typed,
            raw_arguments=dict(arguments),
        )

    def _get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolContractError(f"Неизвестный tool {name}") from error

    @staticmethod
    def _validate_schema(
        arguments: dict[str, Any],
        validator: Draft202012Validator,
    ) -> None:
        try:
            validator.validate(arguments)
        except ValidationError as error:
            location = ".".join(str(part) for part in error.absolute_path)
            suffix = f" по пути {location}" if location else ""
            raise ToolContractError(
                f"Аргументы tool call не соответствуют JSON Schema{suffix}: {error.message}"
            ) from error
