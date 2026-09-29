from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.api.routes.datasets import router as datasets_router


app = FastAPI(
    title="Autonomous Data Readiness Platform",
    description=(
        "API for autonomous dataset profiling, quality analysis, "
        "preparation, validation, relationship discovery, and readiness assessment."
    ),
    version="1.0.0",
)


app.include_router(health_router)
app.include_router(datasets_router)


@app.get("/")
def root():
    return {
        "name": "Autonomous Data Readiness Platform",
        "status": "running",
        "docs": "/docs",
    }