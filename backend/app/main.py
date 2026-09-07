from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .datasets.router import router
from .datasets.store import DatasetStore
from .profiling.profiler import ProfileService


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
    app.include_router(router)
    return app


# Run with: uvicorn app.main:create_app --factory
