from fastapi import FastAPI

from app import __version__
from app.api.routes import documents, health, search
from app.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.embedding import Embedder, get_shared_embedder
from app.repository import InMemoryDocumentStore
from app.retrieval import NumpyVectorIndex


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Find the signal in your documents.",
    )
    app.state.settings = settings
    app.state.document_store = InMemoryDocumentStore()
    app.state.embedder = embedder or get_shared_embedder(settings.embedding_model)
    app.state.vector_index = NumpyVectorIndex()
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router, prefix=settings.api_v1_prefix)
    app.include_router(search.router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
