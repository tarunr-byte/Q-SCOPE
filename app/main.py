"""Canonical FastAPI application and application-lifecycle management for Q-SCOPE.

Handles database initialization, 7-agent pipeline lifecycle, and serves the
unified liquid-glass frontend directly on Railway and local environments.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.graph import build_qscope_graph
from agents.models import close_model_client
from app.api.routes import router
from app.core.config import settings
from app.core.database import Base, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize database tables and pipeline engine at startup."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Q-SCOPE Database schema ready.")
    except Exception as exc:
        logger.error("Failed to auto-create database tables: %s", exc)

    app.state.qscope_engine = build_qscope_graph()
    logger.info("Q-SCOPE API & 7-Agent Pipeline startup complete.")
    try:
        yield
    finally:
        await close_model_client()
        logger.info("Q-SCOPE API shutdown complete.")


app = FastAPI(
    title="Q-SCOPE Operations Studio",
    description=(
        "Unified seven-agent quantum telemetry analysis studio with deterministic integrity, "
        "phase estimation, live tracking journey, and Azhva cognitive translation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PUT"],
    allow_headers=["*"],
)
app.include_router(router)

# Mount the static frontend directory.
# This serves index.html, static images, and styles directly at root (/).
FRONTEND_DIR = settings.project_root / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info("Mounted frontend static files from: %s", FRONTEND_DIR)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected faults privately and return a non-sensitive error response."""
    logger.exception("Unhandled API exception path=%s (%s)", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "request_id": request.headers.get("x-request-id")},
    )
