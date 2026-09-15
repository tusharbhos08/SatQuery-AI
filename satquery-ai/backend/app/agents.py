"""SatQuery AI — Agent System

Three agents:
1. Input Validation Agent — inspects uploaded files, detects modality, checks compatibility
2. Query Understanding Agent — classifies natural-language queries into task types
3. Task Planner / Orchestrator — selects and executes appropriate specialist tools
"""
import re
import time
import logging
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
from app.geospatial import (
    read_image_metadata, GeoTIFFMetadata, check_pair_compatibility
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Task Types
# ══════════════════════════════════════════════════════════════════════════════

class TaskType(str, Enum):
    VQA = "VQA"
    CAPTIONING = "CAPTIONING"
    GROUNDING = "GROUNDING"
    WATER_DETECTION = "WATER_DETECTION"
    BUILDING_DETECTION = "BUILDING_DETECTION"
    ROAD_DETECTION = "ROAD_DETECTION"
    VEGETATION_ANALYSIS = "VEGETATION_ANALYSIS"
    LAND_COVER_CLASSIFICATION = "LAND_COVER_CLASSIFICATION"
    CHANGE_DETECTION = "CHANGE_DETECTION"
    CHANGE_VQA = "CHANGE_VQA"
    OPTICAL_SAR_ANALYSIS = "OPTICAL_SAR_ANALYSIS"
    SPATIAL_RELATIONSHIP = "SPATIAL_RELATIONSHIP"
    MULTI_STEP_ANALYSIS = "MULTI_STEP_ANALYSIS"


# ══════════════════════════════════════════════════════════════════════════════
#  Execution Trace
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TraceStep:
    step: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    time_ms: float = 0
    details: str = ""

    def to_dict(self):
        return {
            "step": self.step,
            "status": self.status,
            "time_ms": round(self.time_ms, 1),
            "details": self.details,
        }


@dataclass
class ExecutionTrace:
    steps: list[TraceStep] = field(default_factory=list)
    total_time_ms: float = 0

    def add(self, step: str, status: str = "completed", time_ms: float = 0, details: str = ""):
        self.steps.append(TraceStep(step=step, status=status, time_ms=time_ms, details=details))

    def to_dict(self):
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_time_ms": round(self.total_time_ms, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Agent 1 — Input Validation Agent
# ══════════════════════════════════════════════════════════════════════════════

class InputValidationAgent:
    """Inspects uploaded files, extracts metadata, detects modality, checks compatibility."""

    SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.geotiff', '.png', '.jpg', '.jpeg'}

    def validate_single(self, filepath: str) -> dict:
        """Validate a single image file."""
        t0 = time.time()
        from pathlib import Path
        path = Path(filepath)

        result = {
            "valid": False,
            "filepath": filepath,
            "filename": path.name,
            "extension": path.suffix.lower(),
            "metadata": {},
            "modality": "unknown",
            "warnings": [],
            "errors": [],
        }

        # Extension check
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            result["errors"].append(f"Unsupported file format: {path.suffix}")
            return result

        # File existence
        if not path.exists():
            result["errors"].append("File not found")
            return result

        # File size check
        file_size = path.stat().st_size
        if file_size == 0:
            result["errors"].append("Empty file")
            return result

        # Read metadata
        meta = read_image_metadata(filepath)
        result["metadata"] = meta.to_dict()
        result["warnings"] = meta.warnings

        # Dimension check
        if meta.width == 0 or meta.height == 0:
            result["errors"].append("Could not determine image dimensions")
            return result

        if meta.width > 20000 or meta.height > 20000:
            result["warnings"].append(f"Very large image ({meta.width}x{meta.height}) — will be tiled for processing")

        # Modality detection
        result["modality"] = self._detect_modality(meta, filepath)

        result["valid"] = len(result["errors"]) == 0
        result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)

        return result

    def validate_pair(self, filepath_a: str, filepath_b: str) -> dict:
        """Validate a pair of images for bi-temporal or cross-modal analysis."""
        val_a = self.validate_single(filepath_a)
        val_b = self.validate_single(filepath_b)

        result = {
            "valid": val_a["valid"] and val_b["valid"],
            "image_count": 2,
            "image_a": val_a,
            "image_b": val_b,
            "modalities": [val_a["modality"], val_b["modality"]],
            "pair_type": "unknown",
            "same_area": True,
            "compatibility": {},
            "warnings": [],
        }

        if not result["valid"]:
            return result

        # Check compatibility
        meta_a = read_image_metadata(filepath_a)
        meta_b = read_image_metadata(filepath_b)
        compat = check_pair_compatibility(meta_a, meta_b)
        result["compatibility"] = compat
        result["same_area"] = compat["geographic_overlap"]
        result["warnings"] = compat["warnings"]

        # Determine pair type
        mod_a, mod_b = val_a["modality"], val_b["modality"]
        if mod_a != mod_b and {mod_a, mod_b} & {"sar"}:
            result["pair_type"] = "cross_modal"
        elif mod_a == mod_b:
            result["pair_type"] = "bitemporal"
        else:
            result["pair_type"] = "mixed"

        # CRS
        if meta_a.crs != "unknown":
            result["crs"] = meta_a.crs
        elif meta_b.crs != "unknown":
            result["crs"] = meta_b.crs

        # Resolution
        if meta_a.resolution:
            result["resolution"] = meta_a.resolution

        return result

    def _detect_modality(self, meta: GeoTIFFMetadata, filepath: str) -> str:
        """Heuristic modality detection: optical, sar, multispectral."""
        filename = filepath.lower()

        # Filename hints
        if any(k in filename for k in ['sar', 'radar', 'risat', 'sentinel1', 's1']):
            return "sar"
        if any(k in filename for k in ['optical', 'cartosat', 'sentinel2', 's2', 'landsat']):
            return "optical"

        # Band count heuristics
        if meta.bands == 1:
            # Single band — could be SAR or panchromatic
            return "sar"  # conservative: assume SAR for single band
        elif meta.bands <= 3:
            return "optical"
        elif meta.bands >= 4:
            return "multispectral"

        return "optical"


# ══════════════════════════════════════════════════════════════════════════════
#  Agent 2 — Query Understanding Agent
# ══════════════════════════════════════════════════════════════════════════════

class QueryUnderstandingAgent:
    """Classifies natural-language queries into structured task types."""

    # Keyword patterns for task classification
    PATTERNS = {
        TaskType.WATER_DETECTION: [
            r'\bwater\b', r'\briver\b', r'\blake\b', r'\bpond\b', r'\bflood\b',
            r'\bocean\b', r'\bsea\b', r'\breservoir\b', r'\bwater.?bod',
        ],
        TaskType.BUILDING_DETECTION: [
            r'\bbuilding\b', r'\bbuilt.?up\b', r'\bconstruct', r'\burban\b',
            r'\bsettlement\b', r'\bhouse\b', r'\bstructure\b',
        ],
        TaskType.ROAD_DETECTION: [
            r'\broad\b', r'\bhighway\b', r'\bstreet\b', r'\bpath\b',
            r'\broute\b', r'\btransport',
        ],
        TaskType.VEGETATION_ANALYSIS: [
            r'\bvegetation\b', r'\bndvi\b', r'\bgreen\b', r'\bforest\b',
            r'\btree\b', r'\bcrop\b', r'\bplant\b', r'\bdeforest',
        ],
        TaskType.LAND_COVER_CLASSIFICATION: [
            r'\bland.?cover\b', r'\bclassif', r'\bland.?use\b', r'\bcategor',
        ],
        TaskType.CHANGE_DETECTION: [
            r'\bchang\w*\b', r'\bbefore\b.*\bafter\b', r'\btemporal\b',
            r'\bdifference\b', r'\bcompar', r'\bevolut',
        ],
        TaskType.GROUNDING: [
            r'\bhighlight\b', r'\blocate\b', r'\bfind\b', r'\bshow\b.*\bwhere\b',
            r'\bidentif\w*\b.*\bregion', r'\bbounding\b', r'\bsegment\b',
        ],
        TaskType.CAPTIONING: [
            r'\bdescrib\w*\b', r'\bcaption\b', r'\bwhat\b.*\bsee\b',
            r'\btell\b.*\babout\b', r'\bexplain\b.*\bimage\b',
            r'\bwhat\b.*\bvisible\b', r'\bscene\b',
        ],
        TaskType.OPTICAL_SAR_ANALYSIS: [
            r'\bsar\b', r'\bradar\b', r'\bfusion\b', r'\bcross.?modal\b',
            r'\boptical\b.*\bsar\b', r'\bsar\b.*\boptical\b',
            r'\bboth\b.*\bimage', r'\bcombine\b',
        ],
    }

    def classify(self, query: str, image_count: int = 1, pair_type: str = "single",
                 modalities: list[str] = None) -> dict:
        """Classify a user query into a task type with confidence."""
        query_lower = query.lower().strip()
        modalities = modalities or ["optical"]

        scores: dict[TaskType, float] = {}

        # Pattern matching
        for task_type, patterns in self.PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                score += len(matches) * 1.0
            if score > 0:
                scores[task_type] = score

        # Explicit Grounding / Localization Imperatives
        if re.search(r'\b(?:highlight|locate|box|bounding box|show where)\b', query_lower):
            scores[TaskType.GROUNDING] = scores.get(TaskType.GROUNDING, 0) + 3.0

        # Context-based adjustments
        if image_count >= 2 and pair_type == "bitemporal":
            scores[TaskType.CHANGE_DETECTION] = scores.get(TaskType.CHANGE_DETECTION, 0) + 3.0
            scores[TaskType.CHANGE_VQA] = scores.get(TaskType.CHANGE_VQA, 0) + 2.0

        if pair_type == "cross_modal" or "sar" in modalities:
            scores[TaskType.OPTICAL_SAR_ANALYSIS] = scores.get(TaskType.OPTICAL_SAR_ANALYSIS, 0) + 3.0

        # Multi-step detection requires explicit spatial constraints (distance/buffer or spatial relation between 2+ features)
        has_distance_or_buffer = bool(re.search(r'\b(?:within|buffer|radius|distance of)\s+\d+\s*(?:m|meters?|km|kilometers?)\b', query_lower))
        has_spatial_rel = bool(re.search(r'\b(?:within|near|close to|adjacent to|proximity of)\b', query_lower))
        
        feature_tasks = [t for t in scores if t in (
            TaskType.WATER_DETECTION, TaskType.BUILDING_DETECTION, 
            TaskType.ROAD_DETECTION, TaskType.VEGETATION_ANALYSIS, 
            TaskType.CHANGE_DETECTION
        )]
        
        if (has_distance_or_buffer and len(feature_tasks) >= 1) or (has_spatial_rel and len(feature_tasks) >= 2):
            scores[TaskType.MULTI_STEP_ANALYSIS] = 10.0

        # Spatial relationship between features (not temporal between dates)
        if re.search(r'\b(?:near|adjacent to|next to|along)\b|\bbetween\s+(?:the\s+)?(?:water|building|road|forest|river)\b', query_lower) and TaskType.MULTI_STEP_ANALYSIS not in scores:
            scores[TaskType.SPATIAL_RELATIONSHIP] = scores.get(TaskType.SPATIAL_RELATIONSHIP, 0) + 1.5

        # Strong change intent
        if re.search(r'\b(?:what changed|how did it change|change between|temporal difference)\b', query_lower):
            scores[TaskType.CHANGE_DETECTION] = scores.get(TaskType.CHANGE_DETECTION, 0) + 4.0

        # Default to VQA/captioning
        if not scores:
            if '?' in query or any(w in query_lower for w in ['what', 'how', 'where', 'which', 'is there']):
                scores[TaskType.VQA] = 1.0
            else:
                scores[TaskType.CAPTIONING] = 1.0

        # Select top task
        if scores:
            sorted_tasks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            primary_task = sorted_tasks[0][0]
            max_score = sorted_tasks[0][1]
            total_score = sum(s for _, s in sorted_tasks)
            confidence = min(0.95, max_score / (total_score + 1.0) * 0.8 + 0.3)
        else:
            primary_task = TaskType.VQA
            confidence = 0.5

        # Build sub-tasks for multi-step
        sub_tasks = []
        if primary_task == TaskType.MULTI_STEP_ANALYSIS:
            for task_type, score in sorted_tasks:
                if task_type != TaskType.MULTI_STEP_ANALYSIS and score > 0:
                    sub_tasks.append(task_type.value)

        return {
            "task_type": primary_task.value,
            "confidence": round(confidence, 3),
            "query": query,
            "sub_tasks": sub_tasks,
            "all_scores": {t.value: round(s, 2) for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)},
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Agent 3 — Task Planner / Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class TaskOrchestrator:
    """Selects and executes specialist tools based on classified task."""

    def __init__(self, model_registry):
        self.registry = model_registry
        self.input_validator = InputValidationAgent()
        self.query_agent = QueryUnderstandingAgent()

    def plan_and_execute(
        self,
        query: str,
        image_paths: list[str],
        images: list[np.ndarray],
        metadata: list[dict],
    ) -> dict:
        """Full agentic pipeline: validate → classify → plan → execute → aggregate."""
        trace = ExecutionTrace()
        t_start = time.time()

        # ── Step 1: Input Validation ──────────────────────────────────────
        t0 = time.time()
        validations = []
        modalities = []
        for path in image_paths:
            v = self.input_validator.validate_single(path)
            validations.append(v)
            modalities.append(v.get("modality", "optical"))

        pair_type = "single"
        if len(image_paths) == 2:
            pair_val = self.input_validator.validate_pair(image_paths[0], image_paths[1])
            pair_type = pair_val.get("pair_type", "bitemporal")

        trace.add("Input validation", "completed",
                  (time.time() - t0) * 1000,
                  f"{len(image_paths)} image(s), modalities: {modalities}")

        for i, v in enumerate(validations):
            mod = v.get("modality", "unknown")
            trace.add(f"Modality detected (image {i+1}): {mod.upper()}", "completed")

        # ── Step 2: Query Classification ──────────────────────────────────
        t0 = time.time()
        classification = self.query_agent.classify(
            query, len(image_paths), pair_type, modalities
        )
        task_type = classification["task_type"]
        trace.add(f"Query classified: {task_type}", "completed",
                  (time.time() - t0) * 1000,
                  f"confidence: {classification['confidence']:.1%}")

        # ── Step 3: Plan Execution ────────────────────────────────────────
        plan = self._create_plan(task_type, classification, pair_type, modalities)
        for step in plan:
            trace.add(f"Planned: {step}", "completed")

        # ── Step 4: Execute Pipeline ──────────────────────────────────────
        t0 = time.time()
        from app.pipelines import PipelineExecutor
        executor = PipelineExecutor(self.registry)

        result = executor.execute(
            task_type=task_type,
            query=query,
            images=images,
            image_paths=image_paths,
            metadata=metadata,
            pair_type=pair_type,
            modalities=modalities,
            classification=classification,
            trace=trace,
        )

        trace.add("Pipeline completed", "completed", (time.time() - t0) * 1000)

        # ── Step 5: Aggregate ─────────────────────────────────────────────
        trace.total_time_ms = (time.time() - t_start) * 1000
        trace.add("Response generated", "completed")

        result["execution_trace"] = trace.to_dict()
        result["task_type"] = task_type
        result["classification"] = classification
        result["processing_time_ms"] = trace.total_time_ms

        return result

    def _create_plan(self, task_type: str, classification: dict,
                     pair_type: str, modalities: list[str]) -> list[str]:
        """Create a human-readable execution plan."""
        plans = {
            "VQA": [
                "Load and preprocess image",
                "Run VQA model",
                "Generate evidence",
                "Calculate confidence",
            ],
            "CAPTIONING": [
                "Load and preprocess image",
                "Analyze scene composition",
                "Generate scene description",
                "Calculate confidence",
            ],
            "GROUNDING": [
                "Load and preprocess image",
                "Identify target concept",
                "Run grounding/segmentation model",
                "Generate bounding box + mask overlay",
                "Calculate confidence",
            ],
            "WATER_DETECTION": [
                "Load and preprocess image",
                "Run water segmentation model",
                "Refine mask",
                "Calculate connected components",
                "Estimate areas",
                "Generate visual overlay",
            ],
            "BUILDING_DETECTION": [
                "Load and preprocess image",
                "Run building detection model",
                "Extract building footprints",
                "Count buildings and estimate area",
                "Generate visual overlay",
            ],
            "ROAD_DETECTION": [
                "Load and preprocess image",
                "Run road extraction model",
                "Skeletonize road network",
                "Estimate road length",
                "Generate visual overlay",
            ],
            "VEGETATION_ANALYSIS": [
                "Load and preprocess image",
                "Check for NIR band availability",
                "Compute vegetation index",
                "Classify vegetation health",
                "Generate vegetation map",
            ],
            "LAND_COVER_CLASSIFICATION": [
                "Load and preprocess image",
                "Run land cover classifier",
                "Generate classification map",
                "Calculate class percentages",
                "Generate legend",
            ],
            "CHANGE_DETECTION": [
                "Validate bi-temporal pair",
                "Align images",
                "Run change detection model",
                "Generate change map",
                "Classify changes",
                "Calculate change statistics",
            ],
            "CHANGE_VQA": [
                "Validate bi-temporal pair",
                "Align images",
                "Run change detection model",
                "Analyze change patterns",
                "Generate change explanation",
            ],
            "OPTICAL_SAR_ANALYSIS": [
                "Validate optical-SAR pair",
                "Extract optical features",
                "Extract SAR features",
                "Execute feature fusion",
                "Generate fused classification",
                "Generate visual evidence",
            ],
            "MULTI_STEP_ANALYSIS": [],
        }

        plan = plans.get(task_type, ["Execute analysis pipeline"])

        # Add multi-step sub-tasks
        if task_type == "MULTI_STEP_ANALYSIS" and classification.get("sub_tasks"):
            for i, sub in enumerate(classification["sub_tasks"], 1):
                sub_plan = plans.get(sub, [f"Execute {sub}"])
                plan.append(f"Step {i}: {sub}")
                for sp in sub_plan[:2]:
                    plan.append(f"  → {sp}")
            plan.append("Spatial intersection")
            plan.append("Generate combined evidence")

        return plan
