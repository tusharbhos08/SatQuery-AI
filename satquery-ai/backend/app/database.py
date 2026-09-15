"""SatQuery AI — Database Models (SQLAlchemy + SQLite)"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, JSON,
    ForeignKey, create_engine
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship

from app.config import settings


class Base(DeclarativeBase):
    pass


# ── Users ──────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), default="")
    role = Column(String(50), default="user")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)


# ── Images ─────────────────────────────────────────────────────────────────────
class ImageRecord(Base):
    __tablename__ = "images"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    filename = Column(String(500), nullable=False)
    filepath = Column(String(1000), nullable=False)
    file_size = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    bands = Column(Integer, default=3)
    modality = Column(String(50))  # optical, sar, multispectral
    crs = Column(String(100))
    resolution = Column(Float)
    bounds = Column(JSON)
    acquisition_date = Column(String(100))
    metadata_json = Column(JSON)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Analysis Requests ──────────────────────────────────────────────────────────
class AnalysisRequest(Base):
    __tablename__ = "analysis_requests"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    query = Column(Text, nullable=False)
    task_type = Column(String(100))
    image_ids = Column(JSON)  # list of image IDs
    pair_type = Column(String(50))  # single, bitemporal, cross_modal
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    result = Column(JSON)
    confidence = Column(Float)
    execution_trace = Column(JSON)
    models_used = Column(JSON)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


# ── Model Registry ─────────────────────────────────────────────────────────────
class ModelRecord(Base):
    __tablename__ = "model_registry"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), unique=True, nullable=False)
    version = Column(String(50), default="1.0")
    task = Column(String(100))
    input_type = Column(String(100))
    output_type = Column(String(100))
    supported_modalities = Column(JSON)
    confidence_support = Column(Boolean, default=True)
    device = Column(String(20), default="cpu")
    status = Column(String(50), default="ready")
    description = Column(Text, default="")
    parameters = Column(JSON)


# ── Reports ────────────────────────────────────────────────────────────────────
class Report(Base):
    __tablename__ = "reports"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("analysis_requests.id"))
    format = Column(String(20))  # html, json, pdf
    filepath = Column(String(1000))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Training Runs ──────────────────────────────────────────────────────────────
class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(200))
    dataset = Column(String(200))
    method = Column(String(100))
    config = Column(JSON)
    status = Column(String(50), default="pending")
    epochs_completed = Column(Integer, default=0)
    total_epochs = Column(Integer, default=10)
    train_loss = Column(Float)
    val_score = Column(Float)
    checkpoint_path = Column(String(1000))
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


# ── Database Engine ────────────────────────────────────────────────────────────
engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency for getting DB session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
