"""Generates the small test files under tests/data.

Invoked by the session fixture in conftest.py (idempotent, regenerated every
run) or directly: python tests/data/make_fixtures.py
"""

from datetime import datetime
from pathlib import Path

import polars as pl

BASIC_CSV = (
    "ts,city,value\n"
    "2024-01-01 08:00:00,Taipei,12.5\n"
    "2024-01-02 08:00:00,Kaohsiung,\n"
    "2024-01-03 08:00:00,Taipei,30.1\n"
)


def generate_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "basic.csv").write_text(BASIC_CSV)
    (out_dir / "latin1.csv").write_bytes("name,val\nCafé,1\nNiño,2\n".encode("latin-1"))
    (out_dir / "nan_inf.csv").write_text("v\n1.5\nNaN\ninf\n-inf\n")
    (out_dir / "empty.csv").write_bytes(b"")
    (out_dir / "header_only.csv").write_text("a,b\n")
    (out_dir / "dup_blank_cols.csv").write_text("a,a,\n1,2,3\n4,5,6\n")
    (out_dir / "corrupt.parquet").write_bytes(b"not a parquet file")
    (out_dir / "notes.txt").write_text("plain text, unsupported format\n")

    pl.DataFrame({"city": ["Taipei", "Tainan"], "pop": [2_600_000, 1_870_000]}).write_excel(
        out_dir / "basic.xlsx"
    )
    pl.DataFrame(
        {"ts": [datetime(2024, 1, 1, 8), datetime(2024, 1, 2, 8)], "value": [1.0, None]}
    ).write_parquet(out_dir / "basic.parquet")


if __name__ == "__main__":
    generate_all(Path(__file__).resolve().parent)
