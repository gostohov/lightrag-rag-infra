class DomainError(Exception):
    """Базовая ошибка предметной модели Workbench."""


class DomainValidationError(DomainError):
    """Нарушен инвариант предметной сущности."""


class EvidenceScopeError(DomainValidationError):
    """Доказательство используется вне разрешённой области."""


class PathNotAllowedError(DomainValidationError):
    """Операция пытается изменить запрещённое поле карточки."""
