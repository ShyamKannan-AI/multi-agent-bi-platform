from pathlib import Path

from src.ingestion.loaders import SUPPORTED_EXTENSIONS


def validate_file(file_path: str | Path) -> None:
    """
    Validate that the input file exists and uses a supported format.
    """

    path = Path(file_path)

    # Check file format first
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Then check whether the file exists
    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )