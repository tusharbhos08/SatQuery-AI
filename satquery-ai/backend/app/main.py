"""SatQuery AI — FastAPI Application Entrypoint"""
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure backend directory is in sys.path for portable execution from any CWD
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("satquery")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure upload/report/demo directories exist
    logger.info("Initializing SatQuery AI Backend...")
    for d in [
        settings.upload_dir,
        f"{settings.upload_dir}/raw",
        f"{settings.upload_dir}/evidence",
        settings.report_dir,
        settings.demo_dir,
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    try:
        await init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.warning(f"Database init warning: {e}")

    yield
    logger.info("Shutting down SatQuery AI Backend...")


app = FastAPI(
    title="SatQuery AI",
    description="Agentic Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis (SIH26167)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Next.js / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Directories for serving processed evidence, reports, demo images
uploads_path = Path(settings.upload_dir)
uploads_path.mkdir(parents=True, exist_ok=True)
app.mount("/api/static/evidence", StaticFiles(directory=str(uploads_path / "evidence")), name="evidence")

reports_path = Path(settings.report_dir)
reports_path.mkdir(parents=True, exist_ok=True)
app.mount("/api/static/reports", StaticFiles(directory=str(reports_path)), name="reports")

demo_path = Path(settings.demo_dir)
demo_path.mkdir(parents=True, exist_ok=True)
app.mount("/api/static/demo", StaticFiles(directory=str(demo_path)), name="demo")

# Include main API router
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "SatQuery AI",
        "tagline": "Ask Questions. Understand Earth.",
        "version": "1.0.0",
        "docs": "/docs",
        "api_prefix": "/api",
        "status": "online",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.backend_host, port=settings.backend_port, reload=True)
