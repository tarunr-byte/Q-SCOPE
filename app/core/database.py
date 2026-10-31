"""Database connection and session management for Q-SCOPE.

Handles PostgreSQL connection on Railway (via DATABASE_URL) with seamless SQLite fallback.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)

db_url = settings.database_url
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

is_sqlite = db_url.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

try:
    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
    logger.info("Q-SCOPE Database connected: %s", "SQLite (local)" if is_sqlite else "PostgreSQL (remote)")
except Exception as exc:
    logger.warning("Primary database connection failed (%s). Falling back to SQLite.", exc)
    fallback_path = settings.project_root / "qscope.db"
    db_url = f"sqlite:///{fallback_path.as_posix()}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator:
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
