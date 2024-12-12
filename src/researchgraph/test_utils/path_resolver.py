import os
from pathlib import Path
from typing import Optional

class TestPathResolver:
    """Utility class for resolving test paths consistently across environments."""

    def __init__(self):
        self.github_workspace = os.environ.get("GITHUB_WORKSPACE")
        # If we're in the repo directory, use that as base_dir
        if os.path.exists(os.path.join(os.getcwd(), "src", "researchgraph")):
            self.base_dir = os.getcwd()
        # If we're in a subdirectory of the repo, find the repo root
        elif os.path.exists(os.path.join(os.path.dirname(os.getcwd()), "src", "researchgraph")):
            self.base_dir = os.path.dirname(os.getcwd())
        # Fallback to GitHub workspace or parent directory
        else:
            self.base_dir = self.github_workspace if self.github_workspace else os.path.abspath(os.path.join(os.getcwd(), ".."))
        self.save_dir = os.environ.get("SAVE_DIR", os.path.join(self.base_dir, "test", "outputs"))

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

    def get_save_dir(self) -> str:
        """Get the base directory for saving test outputs."""
        return self.save_dir

    def get_test_file_path(self, filename: str, subdir: Optional[str] = None) -> str:
        """Get full path for a test file, optionally within a subdirectory."""
        if subdir:
            path = os.path.join(self.save_dir, subdir, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path
        return os.path.join(self.save_dir, filename)

    def get_template_dir(self, template_name: str) -> str:
        """Get path to template directory."""
        template_base = os.path.join(self.base_dir, "src", "researchgraph", "graphs", "ai_scientist", "templates")
        return os.path.join(template_base, template_name)

    def get_figures_dir(self) -> str:
        """Get path to figures directory."""
        return os.path.join(self.base_dir, "images")

# Global instance for convenience
path_resolver = TestPathResolver()
