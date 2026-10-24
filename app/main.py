from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import answer, documents, evaluation, health, search
from app.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.embedding import Embedder, get_shared_embedder
from app.evaluation.store import EvaluationRunStore
from app.llm import LLMProvider, build_llm_provider
from app.repository import SqlDocumentStore
from app.repository.database import build_engine, build_session_factory, create_schema
from app.retrieval import NumpyVectorIndex, rebuild_index

logger = logging.getLogger("prism")


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    store: SqlDocumentStore | None = None,
    llm: LLMProvider | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    if store is None:
        engine = build_engine(settings.database_url)
        session_factory = build_session_factory(engine)
        store = SqlDocumentStore(session_factory)
    else:
        engine = store.engine

    if llm is None:
        llm = build_llm_provider(settings)

    vector_index = NumpyVectorIndex()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.auto_create_schema:
            create_schema(engine)
        loaded = store.load_all_with_embeddings()
        indexed = rebuild_index(vector_index, loaded, expected_model=settings.embedding_model)
        if indexed:
            logger.info("Rebuilt vector index with %d chunks from the database.", indexed)
        yield
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        engine.dispose()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Find the signal in your documents.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    app.state.settings = settings
    app.state.document_store = store
    app.state.embedder = embedder or get_shared_embedder(settings.embedding_model)
    app.state.vector_index = vector_index
    app.state.llm = llm
    app.state.evaluation_store = EvaluationRunStore(settings.evaluation_results_dir)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router, prefix=settings.api_v1_prefix)
    app.include_router(search.router, prefix=settings.api_v1_prefix)
    app.include_router(answer.router, prefix=settings.api_v1_prefix)
    app.include_router(evaluation.router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
