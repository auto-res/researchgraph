from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class TextSplitterConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n"
    length_function: callable = len


class RecursiveCharacterTextSplitter:
    """
    A text splitter that recursively splits text into chunks while preserving semantic boundaries.
    """

    def __init__(self, config: Optional[TextSplitterConfig] = None):
        """
        Initialize the text splitter with optional configuration.

        Args:
            config: Optional configuration for chunk size, overlap, and separator
        """
        self.config = config or TextSplitterConfig()

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks while trying to preserve semantic boundaries.

        Args:
            text: Input text to split

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # If text is shorter than chunk size, return it as is
        if self.config.length_function(text) <= self.config.chunk_size:
            return [text]

        # Split on separator
        splits = text.split(self.config.separator)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_length = self.config.length_function(split)

            # If a single split is larger than chunk size, recursively split it
            if split_length > self.config.chunk_size:
                # If we have accumulated content, add it as a chunk
                if current_chunk:
                    chunks.append(self.config.separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Recursively split the large section
                subsplits = self._split_large_section(split)
                chunks.extend(subsplits)
                continue

            # Check if adding this split would exceed chunk size
            if current_length + split_length > self.config.chunk_size and current_chunk:
                chunks.append(self.config.separator.join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(split)
            current_length += split_length

        # Add any remaining content
        if current_chunk:
            chunks.append(self.config.separator.join(current_chunk))

        # Handle overlap
        if self.config.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._create_overlapping_chunks(chunks)

        return chunks

    def _split_large_section(self, text: str) -> List[str]:
        """
        Split a section of text that's larger than the chunk size.
        Uses sentence boundaries where possible.
        """
        # Try to split on sentence boundaries first
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            return self.split_text(self.config.separator.join(sentences))

        # If we can't split on sentences, split on words
        words = text.split()
        if len(words) > 1:
            return self.split_text(" ".join(words))

        # If we can't split on words, split on characters
        return [
            text[i : i + self.config.chunk_size]
            for i in range(0, len(text), self.config.chunk_size)
        ]

    def _create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """
        Create overlapping chunks from the input chunks.
        """
        result: List[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_size = min(
                    self.config.chunk_overlap, self.config.length_function(prev_chunk)
                )
                if overlap_size > 0:
                    chunk = prev_chunk[-overlap_size:] + self.config.separator + chunk
            result.append(chunk)
        return result
