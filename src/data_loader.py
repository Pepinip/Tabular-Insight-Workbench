"""Utilities for loading tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd


MAX_FILE_SIZE_MB = 25
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@dataclass(frozen=True)
class LoadedDataset:
    """Container for a loaded dataset and basic source metadata."""

    dataframe: pd.DataFrame
    filename: str
    extension: str
    dataset_hash: str
    size_mb: float


class DatasetLoadError(ValueError):
    """Raised when a dataset cannot be loaded safely."""


def calculate_hash(file_bytes: bytes) -> str:
    """Return a stable hash for uploaded file contents."""
    return sha256(file_bytes).hexdigest()


def validate_file(filename: str, file_bytes: bytes, max_size_mb: int = MAX_FILE_SIZE_MB) -> None:
    """Validate extension and size before parsing file contents."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DatasetLoadError(f"Formato no soportado. Usa uno de: {supported}.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise DatasetLoadError(f"Archivo demasiado grande: {size_mb:.2f} MB. Limite: {max_size_mb} MB.")

    if len(file_bytes) == 0:
        raise DatasetLoadError("El archivo esta vacio.")


def load_tabular_file(filename: str, file_bytes: bytes) -> LoadedDataset:
    """Load CSV/XLS/XLSX bytes into a pandas DataFrame."""
    validate_file(filename, file_bytes)
    extension = Path(filename).suffix.lower()

    try:
        if extension == ".csv":
            dataframe = pd.read_csv(BytesIO(file_bytes))
        else:
            dataframe = pd.read_excel(BytesIO(file_bytes))
    except Exception as exc:
        raise DatasetLoadError(f"No se pudo leer el archivo: {exc}") from exc

    if dataframe.empty:
        raise DatasetLoadError("El dataset no contiene filas utiles.")

    return LoadedDataset(
        dataframe=dataframe,
        filename=filename,
        extension=extension,
        dataset_hash=calculate_hash(file_bytes),
        size_mb=len(file_bytes) / (1024 * 1024),
    )