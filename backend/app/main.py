from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .charts.render import CastedFrameCache
from .config import Settings
from .datasets.router import router
from .datasets.store import DatasetStore
from .llm.provider import DisabledProvider, OpenAICompatProvider
from .llm.service import RecommendationService
from .profiling.profiler import ProfileService

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="VizPilot API")
    app.add_middleware(
        CORSMiddleware,
        # local dev frontend only (Vite/CRA on any localhost port)
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.store = DatasetStore(settings.data_dir)
    app.state.profiles = ProfileService(settings.data_dir, settings.profile_sample_threshold)
    app.state.render_cache = CastedFrameCache()
    if settings.llm_enabled:
        provider = OpenAICompatProvider(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            include_sample_rows=settings.llm_include_sample_rows,
        )
    else:
        provider = DisabledProvider()
    app.state.recommendations = RecommendationService(provider)
    app.include_router(router)
    # single-port production mode: serve the built SPA when it exists
    # (dev mode uses the vite proxy instead; /api routes are matched first)
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app


# Run with: uvicorn app.main:create_app --factory
