from .main import AIIntegratorv1
from .config import ConfigLoader

# Initialize config loader
config = ConfigLoader()

# Export configuration
ai_integratorv1_setting = config.load_settings()
ai_integrator_v1_extractor_prompt = config.get_prompt("extractor_prompt")
ai_integrator_v1_codeextractor_prompt = config.get_prompt("codeextractor_prompt")
ai_integrator_v1_creator_prompt = config.get_prompt("creator_prompt")

__all__ = [
    "AIIntegratorv1",
    "ai_integrator_v1_extractor_prompt",
    "ai_integrator_v1_codeextractor_prompt",
    "ai_integrator_v1_creator_prompt",
    "ai_integratorv1_setting",
]
