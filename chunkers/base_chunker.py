
# This script is copied/adapted from the original code from https://github.com/brandonstarxel/chunking_evaluation
# License: MIT License

from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass