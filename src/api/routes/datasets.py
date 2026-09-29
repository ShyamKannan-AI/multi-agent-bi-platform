from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.orchestration.workflow import AgenticReadinessWorkflow


router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
}


# One workflow instance is intentionally reused so that
# LangGraph's MemorySaver can preserve thread state.
workflow = AgenticReadinessWorkflow()


def _find_uploaded_dataset(dataset_id: str) -> Path:
    """
    Locate the file belonging to a previously uploaded dataset.

    Uploaded files are stored as:

        data/uploads/<dataset_id>/<filename>
    """

    dataset_directory = UPLOAD_DIR / dataset_id

    if not dataset_directory.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' was not found.",
        )

    files = [
        path
        for path in dataset_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"No uploaded dataset file found for '{dataset_id}'.",
        )

    if len(files) > 1:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Multiple dataset files were found for '{dataset_id}'. "
                "The dataset directory should contain exactly one uploaded file."
            ),
        )

    return files[0]


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{extension}'. "
                f"Allowed types: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    dataset_id = str(uuid4())

    safe_filename = Path(file.filename).name

    dataset_directory = UPLOAD_DIR / dataset_id
    dataset_directory.mkdir(parents=True, exist_ok=True)

    file_path = dataset_directory / safe_filename

    contents = await file.read()

    file_path.write_bytes(contents)

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "filename": safe_filename,
        "file_type": extension,
        "path": str(file_path),
    }


@router.post("/{dataset_id}/analyze")
def analyze_dataset(dataset_id: str):
    """
    Start the autonomous readiness workflow for an uploaded dataset.
    """

    dataset_path = _find_uploaded_dataset(dataset_id)

    thread_id = str(uuid4())

    try:
        result = workflow.run(
            dataset_paths={
                dataset_id: str(dataset_path),
            },
            thread_id=thread_id,
        )

        return {
            "dataset_id": dataset_id,
            "thread_id": thread_id,
            "result": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset analysis failed: {exc}",
        ) from exc