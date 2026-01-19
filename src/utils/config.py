"""Configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2048

    # vLLora Gateway Configuration
    use_local_gateway: bool = False
    llm_base_url: Optional[str] = None  # e.g., http://localhost:9090/v1

    # Stockfish Configuration
    stockfish_path: Optional[str] = None  # Auto-detect if None
    stockfish_depth: int = 15
    stockfish_multipv: int = 3

    # Logging Configuration
    log_level: str = "INFO"

    # Application Settings
    default_user_level: str = "intermediate"
    max_turns_per_session: int = 100

    def get_llm_base_url(self) -> Optional[str]:
        """Get the LLM base URL if local gateway is enabled."""
        if self.use_local_gateway and self.llm_base_url:
            return self.llm_base_url
        return None

    def validate_api_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        # If using local gateway, API key may not be strictly required
        if self.use_local_gateway:
            return True
        return bool(self.openai_api_key and self.openai_api_key != "your-openai-api-key-here")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Default settings instance
settings = get_settings()
