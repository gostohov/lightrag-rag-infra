class StorageError(Exception):
    """Базовая ошибка нового хранилища Workbench."""


class StorageConflictError(StorageError):
    """Запись конфликтует с уже сохранённой ревизией."""


class StorageSchemaError(StorageError):
    """Версия или структура базы несовместима с приложением."""
