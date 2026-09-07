import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile

from ..charts.rules import recommend_charts
from ..config import Settings
from ..serialization import df_to_records
from .loader import SUPPORTED_EXTENSIONS, LoaderError, load_dataframe
from .store import DatasetNotFoundError, DatasetStore

router = APIRouter(prefix="/api")

# Validate before touching the filesystem (stage1 critique #2).
_DATASET_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _store(request: Request) -> DatasetStore:
    return request.app.state.store


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _check_dataset_id(dataset_id: str) -> None:
    if not _DATASET_ID_RE.fullmatch(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/datasets")
async def upload_dataset(request: Request, file: UploadFile) -> dict[str, Any]:
    settings = _settings(request)
    limit_detail = f"File exceeds the {settings.max_upload_mb} MB upload limit"

    # Declared-size check; the multipart body is already spooled to disk by the
    # framework at this point, but this avoids loading oversized files into memory.
    content_length = request.headers.get("content-length", "")
    if content_length.isdigit() and int(content_length) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=limit_detail)

    filename = file.filename or ""
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type; expected one of: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=limit_detail)

    try:
        df = load_dataframe(data, filename)
    except LoaderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _store(request).save(df, filename)


@router.get("/datasets")
def list_datasets(request: Request) -> list[dict[str, Any]]:
    return _store(request).list_meta()


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, request: Request) -> dict[str, Any]:
    _check_dataset_id(dataset_id)
    try:
        return _store(request).get_meta(dataset_id)
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found") from None


@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(
    dataset_id: str,
    request: Request,
    rows: int | None = Query(default=None, ge=1, le=1000),
) -> list[dict[str, Any]]:
    _check_dataset_id(dataset_id)
    try:
        df = _store(request).get_df(dataset_id)
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found") from None
    n = rows if rows is not None else _settings(request).preview_rows
    return df_to_records(df.head(n))


@router.get("/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str, request: Request) -> dict[str, Any]:
    _check_dataset_id(dataset_id)
    try:
        df = _store(request).get_df(dataset_id)
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found") from None
    return request.app.state.profiles.get(dataset_id, df).model_dump()


@router.get("/datasets/{dataset_id}/recommendations")
def get_recommendations(dataset_id: str, request: Request) -> dict[str, Any]:
    _check_dataset_id(dataset_id)
    try:
        df = _store(request).get_df(dataset_id)
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found") from None
    profile = request.app.state.profiles.get(dataset_id, df)
    charts = [rec.model_dump() for rec in recommend_charts(profile)]
    return {
        "charts": charts,
        "insights": [],  # populated by the LLM in Stage 5; shape fixed now
        "message": None if charts else "No charts could be recommended for this dataset.",
    }
