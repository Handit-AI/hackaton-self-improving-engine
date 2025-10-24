"""
Application configuration using Pydantic settings.
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: Optional[str] = None
    
    # OpenAI
    openai_api_key: Optional[str] = None
    
    # ACE System
    enable_darwin_evolution: bool = True  # Enable Darwin-Gödel evolution
    
    # Application
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

