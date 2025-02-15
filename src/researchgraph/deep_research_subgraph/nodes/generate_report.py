from dataclasses import dataclass
from typing import List, Optional, Callable
import asyncio
from datetime import datetime


@dataclass
class ResearchProgress:
    current_depth: int
    total_depth: int
    current_breadth: int
    total_breadth: int
    current_query: Optional[str]
    total_queries: int
    completed_queries: int


class OutputManager:
    """
    Manages console output and progress reporting for the research process.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._progress_callbacks: List[Callable[[ResearchProgress], None]] = []

    async def log(self, *args):
        """Thread-safe logging with timestamp."""
        async with self._lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}]", *args)

    def register_progress_callback(self, callback: Callable[[ResearchProgress], None]):
        """Register a callback for progress updates."""
        self._progress_callbacks.append(callback)

    async def update_progress(self, progress: ResearchProgress):
        """Update research progress and notify all callbacks."""
        async with self._lock:
            for callback in self._progress_callbacks:
                callback(progress)


class ReportGenerator:
    """
    Generates the final markdown report from research results.
    """

    def __init__(self, output_manager: OutputManager):
        self.output = output_manager

    async def generate_report(
        self, prompt: str, learnings: List[str], visited_urls: List[str]
    ) -> str:
        """
        Generate a final markdown report from research results.

        Args:
            prompt: Original research query
            learnings: List of research findings
            visited_urls: List of sources visited

        Returns:
            Markdown formatted report
        """
        # Create report sections
        sections = [
            self._create_header(prompt),
            self._create_summary(learnings),
            self._create_findings(learnings),
            self._create_sources(visited_urls),
        ]

        return "\n\n".join(sections)

    def _create_header(self, prompt: str) -> str:
        """Create report header with title and original query."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"# Research Report\n"
            f"Generated on: {timestamp}\n\n"
            f"## Original Query\n"
            f"{prompt}"
        )

    def _create_summary(self, learnings: List[str]) -> str:
        """Create executive summary from key learnings."""
        if not learnings:
            return "## Summary\nNo significant findings to report."

        # Use first few learnings for summary
        key_points = learnings[:3]
        summary = "\n".join(f"- {point}" for point in key_points)

        return f"## Executive Summary\n" f"{summary}"

    def _create_findings(self, learnings: List[str]) -> str:
        """Create detailed findings section."""
        if not learnings:
            return "## Findings\nNo detailed findings to report."

        findings = "\n".join(f"- {learning}" for learning in learnings)

        return f"## Detailed Findings\n" f"{findings}"

    def _create_sources(self, urls: List[str]) -> str:
        """Create sources section with visited URLs."""
        if not urls:
            return "## Sources\nNo sources to cite."

        sources = "\n".join(f"- {url}" for url in urls)

        return f"## Sources\n" f"{sources}"
