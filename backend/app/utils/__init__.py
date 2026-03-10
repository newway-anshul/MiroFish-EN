"""
Utility module.

This module exposes commonly used utility classes so they can be imported
directly from the `utils` package without needing to reference the internal
module structure.

For example, instead of writing:

    from utils.file_parser import FileParser
    from utils.llm_client import LLMClient

developers can simply write:

    from utils import FileParser, LLMClient

The `__all__` variable explicitly defines the public API of this package,
ensuring that only the intended classes are exported when using:

    from utils import *

Exports:
    FileParser: Utility class responsible for parsing and extracting content
                from various file formats.
    LLMClient:  Client wrapper used for interacting with language models.

This approach keeps imports cleaner and hides internal module structure
from external code.
"""

from .file_parser import FileParser
from .llm_client import LLMClient

__all__ = ['FileParser', 'LLMClient']

