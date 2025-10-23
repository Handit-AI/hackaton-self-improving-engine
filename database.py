"""
Database connection and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import AsyncGenerator

from config import settings


# Check if database URL is configured
if settings.database_url is None:
    raise ValueError(
        "DATABASE_URL environment variable is required. "
        "Please set it in your .env file or environment."
    )

# Ensure database URL uses asyncpg driver
async_database_url = settings.database_url
if async_database_url.startswith("postgresql://"):
    async_database_url = async_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif async_database_url.startswith("postgres://"):
    async_database_url = async_database_url.replace("postgres://", "postgres+asyncpg://", 1)
elif not async_database_url.startswith("postgresql+asyncpg://") and not async_database_url.startswith("postgres+asyncpg://"):
    # If URL doesn't have a driver, add asyncpg
    async_database_url = async_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    async_database_url = async_database_url.replace("postgres://", "postgres+asyncpg://", 1)

# Create async engine
async_engine = create_async_engine(
    async_database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Create async session factory
async_session_maker = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session.
    
    Yields:
        AsyncSession: Database session
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

