"""Configuration management for AI Integrator v1."""
import os
from pathlib import Path
from typing import Dict, Any

import yaml


class ConfigLoader:
    """Configuration loader for AI Integrator v1."""

    def __init__(self, config_dir: str = None):
        """Initialize config loader.

        Args:
            config_dir: Path to config directory. If None, uses default config dir.
        """
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = Path(config_dir)

    def load_prompts(self) -> Dict[str, str]:
        """Load prompt configurations.

        Returns:
            Dict containing prompt configurations.
        """
        prompts_file = self.config_dir / "prompts.yaml"
        with open(prompts_file, "r") as f:
            return yaml.safe_load(f)

    def load_settings(self) -> Dict[str, Any]:
        """Load general settings.

        Returns:
            Dict containing settings.
        """
        settings_file = self.config_dir / "settings.yaml"
        with open(settings_file, "r") as f:
            return yaml.safe_load(f)

    def get_prompt(self, prompt_name: str) -> str:
        """Get a specific prompt by name.

        Args:
            prompt_name: Name of the prompt to retrieve.

        Returns:
            The prompt string.

        Raises:
            KeyError: If prompt_name not found.
        """
        prompts = self.load_prompts()
        if prompt_name not in prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found in configuration")
        return prompts[prompt_name]

    def get_setting(self, setting_name: str) -> Any:
        """Get a specific setting by name.

        Args:
            setting_name: Name of the setting to retrieve.

        Returns:
            The setting value.

        Raises:
            KeyError: If setting_name not found.
        """
        settings = self.load_settings()
        if setting_name not in settings:
            raise KeyError(f"Setting '{setting_name}' not found in configuration")
        return settings[setting_name]
