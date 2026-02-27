"""Custom exceptions for Chicory."""


class ChicoryError(Exception):
    """Base exception for all Chicory errors."""


class DatabaseError(ChicoryError):
    """Database operation failed."""


class MemoryNotFoundError(ChicoryError):
    """Requested memory does not exist."""


class TagNotFoundError(ChicoryError):
    """Requested tag does not exist."""


class EmbeddingError(ChicoryError):
    """Embedding generation or retrieval failed."""


class LLMError(ChicoryError):
    """LLM API call failed."""


class MigrationError(ChicoryError):
    """Model migration operation failed."""
