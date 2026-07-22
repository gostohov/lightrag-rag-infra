from enum import StrEnum


class EpistemicStatus(StrEnum):
    UNKNOWN = "неизвестно"
    NOT_APPLICABLE = "не применимо"
    SOURCE_CONFIRMED = "подтверждено источником"
    ANALYST_CONFIRMED = "подтверждено аналитиком"
    DERIVED = "выведено"


class EvidenceKind(StrEnum):
    SOURCE_FRAGMENT = "фрагмент источника"
    HUMAN_KNOWLEDGE = "экспертное знание"


class EvidenceScope(StrEnum):
    CARD = "карточка"
    SELECTION = "выбранный диапазон"


class GapStatus(StrEnum):
    OPEN = "открыт"
    RESOLVED = "закрыт"
    LEFT_OPEN = "оставлен открытым"


class GapResolutionMode(StrEnum):
    SOURCE_FACT = "source_fact"
    DESIGN_DECISION = "design_decision"
    EXTERNAL_INPUT = "external_input"


class CardDecisionKind(StrEnum):
    INCLUDE = "включить"
    INCLUDE_INCOMPLETE = "включить неполной"
    EXCLUDE = "исключить"
