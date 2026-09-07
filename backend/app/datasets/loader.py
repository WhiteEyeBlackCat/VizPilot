import io
from pathlib import Path

import polars as pl

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".parquet")


class LoaderError(ValueError):
    """Raised for any file that cannot be turned into a usable DataFrame."""


def load_dataframe(data: bytes, filename: str) -> pl.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoaderError(
            f"Unsupported file type '{ext or filename}'; "
            f"expected one of: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    try:
        if ext == ".csv":
            df = _read_csv(data)
        elif ext == ".xlsx":
            df = pl.read_excel(io.BytesIO(data))  # first sheet
        else:
            df = pl.read_parquet(io.BytesIO(data))
    except pl.exceptions.NoDataError:
        raise LoaderError("File is empty (no data)") from None
    except Exception as exc:
        raise LoaderError(f"Failed to parse {ext} file: {exc}") from exc

    if df.width == 0:
        raise LoaderError("File contains no columns")
    if df.height == 0:
        raise LoaderError("File has a header but no data rows")
    return df


def _read_csv(data: bytes) -> pl.DataFrame:
    # data stays in memory so the utf8-lossy retry re-reads the same payload
    try:
        return pl.read_csv(io.BytesIO(data), try_parse_dates=True)
    except pl.exceptions.ComputeError as exc:
        if "utf-8" not in str(exc).lower():
            raise
        return pl.read_csv(io.BytesIO(data), try_parse_dates=True, encoding="utf8-lossy")
