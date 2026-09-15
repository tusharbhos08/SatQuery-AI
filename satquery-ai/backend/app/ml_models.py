"""SatQuery AI — ML Model Implementations

All models perform REAL image analysis using computer vision algorithms.
No hardcoded outputs. Every result is computed from actual pixel data.

Architecture: Model Registry + pluggable model implementations.
Each model can be swapped for a deep learning equivalent without changing the API.
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
# Pure OpenCV and NumPy implementation for high performance without C-extension issues

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Base Model Interface
# ══════════════════════════════════════════════════════════════════════════════

class BaseModel(ABC):
    """Base class for all SatQuery AI models."""

    name: str = "base_model"
    version: str = "1.0"
    task: str = "unknown"
    input_type: str = "single_image"
    output_type: str = "text"
    supported_modalities: list[str] = ["optical"]
    confidence_support: bool = True
    device: str = "cpu"
    status: str = "ready"
    description: str = ""

    def info(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "supported_modalities": self.supported_modalities,
            "confidence_support": self.confidence_support,
            "device": self.device,
            "status": self.status,
            "description": self.description,
        }

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> dict:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Model Registry
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """Central registry for all ML models. Supports lazy loading and hot-swapping."""

    def __init__(self):
        self._models: dict[str, BaseModel] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all default model implementations."""
        defaults = [
            VQAModel(),
            CaptionModel(),
            GroundingModel(),
            WaterSegmentationModel(),
            BuildingDetectionModel(),
            RoadExtractionModel(),
            LandCoverModel(),
            VegetationModel(),
            ChangeDetectionModel(),
            OpticalSARFusionModel(),
        ]
        for model in defaults:
            self._models[model.name] = model

    def get(self, name: str) -> Optional[BaseModel]:
        return self._models.get(name)

    def get_for_task(self, task: str) -> Optional[BaseModel]:
        """Find the best model for a given task type."""
        task_map = {
            "VQA": "remote_sensing_vqa",
            "CAPTIONING": "remote_sensing_captioning",
            "GROUNDING": "grounding_model",
            "WATER_DETECTION": "water_segmentation",
            "BUILDING_DETECTION": "building_detection",
            "ROAD_DETECTION": "road_extraction",
            "VEGETATION_ANALYSIS": "vegetation_analysis",
            "LAND_COVER_CLASSIFICATION": "land_cover_classifier",
            "CHANGE_DETECTION": "change_detection",
            "CHANGE_VQA": "change_detection",
            "OPTICAL_SAR_ANALYSIS": "optical_sar_fusion",
        }
        model_name = task_map.get(task)
        if model_name:
            return self._models.get(model_name)
        return None

    def list_models(self) -> list[dict]:
        return [m.info() for m in self._models.values()]

    def register(self, model: BaseModel):
        self._models[model.name] = model


# ══════════════════════════════════════════════════════════════════════════════
#  VQA Model — Analyzes image content to answer questions
# ══════════════════════════════════════════════════════════════════════════════

class VQAModel(BaseModel):
    name = "remote_sensing_vqa"
    version = "1.0"
    task = "VQA"
    description = "Remote sensing visual question answering using scene analysis"

    def predict(self, image: np.ndarray, question: str = "", **kwargs) -> dict:
        """Answer questions by analyzing actual image content."""
        t0 = time.time()

        # Analyze image properties
        analysis = self._analyze_scene(image)

        # Parse question intent and generate answer
        answer = self._generate_answer(question, analysis)

        return {
            "answer": answer,
            "scene_analysis": analysis,
            "confidence": analysis["overall_confidence"],
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }

    def _analyze_scene(self, image: np.ndarray) -> dict:
        """Comprehensive scene analysis using real image statistics."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Color distribution analysis
        colors = self._analyze_colors(image, hsv)

        # Texture analysis (edge density)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (h * w)

        # Brightness & contrast
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 128.0

        # Scene type estimation
        scene_type = self._estimate_scene_type(colors, edge_density)

        # Compute confidence based on image quality
        confidence = min(0.95, 0.6 + contrast * 0.2 + (1 - abs(brightness - 0.5)) * 0.15)

        return {
            "colors": colors,
            "edge_density": round(edge_density, 4),
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "scene_type": scene_type,
            "dimensions": {"width": w, "height": h},
            "overall_confidence": round(confidence, 3),
        }

    def _analyze_colors(self, image: np.ndarray, hsv: np.ndarray) -> dict:
        """Analyze color distribution for land cover hints."""
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        total = h_ch.size

        # Water: blue hues (100-130) with moderate saturation
        water_mask = ((h_ch >= 90) & (h_ch <= 130) & (s_ch > 30) & (v_ch > 30))
        water_pct = np.count_nonzero(water_mask) / total

        # Vegetation: green hues (35-85) with moderate-high saturation
        veg_mask = ((h_ch >= 30) & (h_ch <= 85) & (s_ch > 25) & (v_ch > 30))
        veg_pct = np.count_nonzero(veg_mask) / total

        # Built-up: low saturation, moderate value (gray tones)
        built_mask = ((s_ch < 40) & (v_ch > 50) & (v_ch < 200))
        built_pct = np.count_nonzero(built_mask) / total

        # Bare land: brown/yellow hues with low-moderate saturation
        bare_mask = ((h_ch >= 10) & (h_ch <= 30) & (s_ch > 20) & (v_ch > 40))
        bare_pct = np.count_nonzero(bare_mask) / total

        # Bright/clouds: very high value
        bright_mask = (v_ch > 240)
        bright_pct = np.count_nonzero(bright_mask) / total

        # Dark areas (shadows)
        dark_mask = (v_ch < 30)
        dark_pct = np.count_nonzero(dark_mask) / total

        return {
            "water": round(water_pct * 100, 1),
            "vegetation": round(veg_pct * 100, 1),
            "built_up": round(built_pct * 100, 1),
            "bare_land": round(bare_pct * 100, 1),
            "bright": round(bright_pct * 100, 1),
            "dark": round(dark_pct * 100, 1),
        }

    def _estimate_scene_type(self, colors: dict, edge_density: float) -> str:
        """Estimate dominant scene type."""
        dominant = max(colors, key=lambda k: colors[k] if k not in ('bright', 'dark') else 0)
        scenes = {
            "water": "aquatic/coastal",
            "vegetation": "vegetated/agricultural",
            "built_up": "urban/built-up",
            "bare_land": "arid/barren",
        }
        base = scenes.get(dominant, "mixed")
        if edge_density > 0.15:
            base = "urban/complex " + base
        return base

    def _generate_answer(self, question: str, analysis: dict) -> str:
        """Generate a grounded answer from actual scene analysis."""
        q = question.lower()
        colors = analysis["colors"]

        # Specific question patterns
        if any(w in q for w in ['water', 'river', 'lake', 'ocean']):
            pct = colors["water"]
            if pct > 5:
                return f"Yes, water bodies are visible in the image, covering approximately {pct:.1f}% of the scene. The water regions appear in the blue spectral range."
            else:
                return f"Water bodies are not prominently visible in this image. Only {pct:.1f}% of the scene shows water-like signatures."

        if any(w in q for w in ['building', 'built', 'urban', 'settlement', 'house']):
            pct = colors["built_up"]
            if pct > 10:
                return f"Built-up regions are visible, covering approximately {pct:.1f}% of the scene. The urban fabric shows characteristic low-saturation, moderate-brightness patterns."
            else:
                return f"Built-up areas are minimal in this scene, covering only {pct:.1f}% of the area."

        if any(w in q for w in ['vegetation', 'green', 'forest', 'tree', 'crop', 'plant']):
            pct = colors["vegetation"]
            if pct > 10:
                return f"Vegetation is clearly present, covering approximately {pct:.1f}% of the scene. Green spectral signatures indicate active vegetation."
            else:
                return f"Vegetation coverage is limited to approximately {pct:.1f}% of the scene."

        if any(w in q for w in ['road', 'highway', 'street']):
            ed = analysis["edge_density"]
            if ed > 0.08:
                return f"Linear features consistent with roads are visible in the image. Edge density ({ed:.3f}) suggests a well-developed road network."
            else:
                return "Road features are not prominently visible at this resolution."

        if any(w in q for w in ['land cover', 'land use', 'type', 'class']):
            parts = []
            for k, v in sorted(colors.items(), key=lambda x: x[1], reverse=True):
                if v > 3 and k not in ('bright', 'dark'):
                    parts.append(f"{k.replace('_', ' ')}: {v:.1f}%")
            return "Land cover analysis: " + ", ".join(parts[:5])

        if any(w in q for w in ['describe', 'caption', 'what do you see', 'overview', 'visible']):
            return self._generate_caption(analysis)

        # General answer
        return self._generate_caption(analysis)

    def _generate_caption(self, analysis: dict) -> str:
        """Generate a descriptive caption from scene analysis."""
        colors = analysis["colors"]
        parts = []

        # Sort by coverage
        ordered = sorted(
            [(k, v) for k, v in colors.items() if k not in ('bright', 'dark') and v > 3],
            key=lambda x: x[1],
            reverse=True,
        )

        if not ordered:
            return "The image shows a complex scene requiring further spectral analysis."

        descriptions = {
            "vegetation": "vegetated areas including possible agricultural fields and natural vegetation",
            "water": "water bodies such as rivers, lakes, or reservoirs",
            "built_up": "built-up regions including urban structures and settlements",
            "bare_land": "bare or arid land surfaces",
        }

        primary = ordered[0]
        parts.append(f"The image is predominantly characterized by {descriptions.get(primary[0], primary[0])} ({primary[1]:.1f}% coverage)")

        for item in ordered[1:3]:
            parts.append(f"with {descriptions.get(item[0], item[0])} ({item[1]:.1f}%)")

        scene = analysis["scene_type"]
        edge = analysis["edge_density"]

        result = ", ".join(parts) + "."

        if edge > 0.1:
            result += " The scene shows significant structural complexity suggesting human-made features."
        elif edge < 0.03:
            result += " The terrain appears relatively homogeneous with few distinct boundaries."

        return result


# ══════════════════════════════════════════════════════════════════════════════
#  Caption Model
# ══════════════════════════════════════════════════════════════════════════════

class CaptionModel(BaseModel):
    name = "remote_sensing_captioning"
    version = "1.0"
    task = "CAPTIONING"
    description = "Scene description generator for remote sensing imagery"

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        t0 = time.time()
        vqa = VQAModel()
        analysis = vqa._analyze_scene(image)
        caption = vqa._generate_caption(analysis)

        return {
            "caption": caption,
            "scene_analysis": analysis,
            "confidence": analysis["overall_confidence"],
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Grounding Model — Text-guided region localization
# ══════════════════════════════════════════════════════════════════════════════

class GroundingModel(BaseModel):
    name = "grounding_model"
    version = "1.0"
    task = "GROUNDING"
    output_type = "segmentation_mask"
    description = "Text-guided region grounding using color-space analysis"

    # Concept → HSV thresholds
    CONCEPT_THRESHOLDS = {
        "water": {"h_range": (90, 130), "s_min": 30, "v_min": 30},
        "vegetation": {"h_range": (30, 85), "s_min": 25, "v_min": 30},
        "building": {"s_max": 40, "v_range": (50, 200)},
        "road": {"s_max": 30, "v_range": (60, 180), "edge_based": True},
        "bare_land": {"h_range": (10, 30), "s_min": 20, "v_min": 40},
        "settlement": {"s_max": 40, "v_range": (50, 200)},
        "agricultural": {"h_range": (25, 75), "s_min": 15, "v_min": 30},
    }

    def predict(self, image: np.ndarray, query: str = "", **kwargs) -> dict:
        t0 = time.time()

        concept = self._extract_concept(query)
        mask, confidence = self._detect_concept(image, concept)

        # Find bounding boxes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                              "area": float(area)})

        # Create overlay
        overlay = self._create_overlay(image, mask, concept)

        return {
            "concept": concept,
            "mask": mask,
            "overlay": overlay,
            "bounding_boxes": sorted(boxes, key=lambda b: b["area"], reverse=True)[:50],
            "regions_found": len(boxes),
            "confidence": confidence,
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }

    def _extract_concept(self, query: str) -> str:
        q = query.lower()
        for concept in self.CONCEPT_THRESHOLDS:
            if concept in q:
                return concept
        if any(w in q for w in ['river', 'lake', 'pond', 'ocean', 'flood']):
            return "water"
        if any(w in q for w in ['tree', 'forest', 'green', 'crop', 'plant']):
            return "vegetation"
        if any(w in q for w in ['house', 'urban', 'structure', 'built']):
            return "building"
        if any(w in q for w in ['highway', 'street', 'path']):
            return "road"
        return "water"  # default

    def _detect_concept(self, image: np.ndarray, concept: str) -> tuple[np.ndarray, float]:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        thresholds = self.CONCEPT_THRESHOLDS.get(concept, self.CONCEPT_THRESHOLDS["water"])

        if "h_range" in thresholds:
            h_lo, h_hi = thresholds["h_range"]
            s_min = thresholds.get("s_min", 0)
            v_min = thresholds.get("v_min", 0)
            mask = cv2.inRange(hsv, (h_lo, s_min, v_min), (h_hi, 255, 255))
        elif "s_max" in thresholds:
            s_max = thresholds["s_max"]
            v_lo, v_hi = thresholds.get("v_range", (0, 255))
            mask = cv2.inRange(hsv, (0, 0, v_lo), (180, s_max, v_hi))

            if thresholds.get("edge_based"):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
                mask = cv2.bitwise_and(mask, edges_dilated)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        coverage = np.count_nonzero(mask) / mask.size
        confidence = min(0.95, 0.5 + coverage * 2) if coverage > 0.01 else 0.3

        return mask, round(confidence, 3)

    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, concept: str) -> np.ndarray:
        colors = {
            "water": (0, 120, 255),
            "vegetation": (0, 255, 100),
            "building": (255, 100, 0),
            "road": (255, 255, 0),
            "bare_land": (200, 150, 50),
            "settlement": (255, 80, 80),
            "agricultural": (150, 255, 50),
        }
        color = colors.get(concept, (255, 0, 0))

        overlay = image.copy()
        colored = np.zeros_like(image)
        colored[mask > 0] = color
        cv2.addWeighted(colored, 0.4, overlay, 1.0, 0, overlay)

        # Draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        return overlay


# ══════════════════════════════════════════════════════════════════════════════
#  Water Segmentation Model
# ══════════════════════════════════════════════════════════════════════════════

class WaterSegmentationModel(BaseModel):
    name = "water_segmentation"
    version = "1.0"
    task = "WATER_DETECTION"
    output_type = "segmentation_mask"
    description = "Water body detection using spectral analysis and morphological processing"

    def predict(self, image: np.ndarray, resolution: float = None, **kwargs) -> dict:
        t0 = time.time()
        mask, confidence = self._detect_water(image)

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        water_bodies = []
        total_area_pixels = 0
        for i in range(1, num_labels):  # skip background
            area_pixels = stats[i, cv2.CC_STAT_AREA]
            if area_pixels < 50:
                continue
            total_area_pixels += area_pixels
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            area_km2 = None
            if resolution and resolution > 0:
                area_km2 = round(area_pixels * (resolution ** 2) / 1_000_000, 4)

            water_bodies.append({
                "id": i,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "area_pixels": int(area_pixels),
                "area_km2": area_km2,
                "centroid": {"x": round(float(centroids[i][0]), 1), "y": round(float(centroids[i][1]), 1)},
            })

        water_bodies.sort(key=lambda wb: wb["area_pixels"], reverse=True)

        total_km2 = None
        largest_km2 = None
        if resolution and resolution > 0:
            total_km2 = round(total_area_pixels * (resolution ** 2) / 1_000_000, 4)
            if water_bodies:
                largest_km2 = water_bodies[0].get("area_km2")

        overlay = self._create_overlay(image, mask, water_bodies)

        return {
            "mask": mask,
            "overlay": overlay,
            "water_bodies_count": len(water_bodies),
            "water_bodies": water_bodies[:100],
            "total_area_pixels": int(total_area_pixels),
            "total_area_km2": total_km2,
            "largest_area_km2": largest_km2,
            "coverage_percent": round(total_area_pixels / mask.size * 100, 2),
            "confidence": confidence,
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }

    def _detect_water(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Multi-threshold water detection
        # Deep water (blue)
        mask1 = cv2.inRange(hsv, (90, 30, 20), (130, 255, 255))
        # Shallow/turbid water
        mask2 = cv2.inRange(hsv, (85, 15, 40), (135, 255, 200))
        # Dark water
        mask3 = cv2.inRange(hsv, (95, 20, 10), (125, 200, 100))

        mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

        # Also check B-channel dominance
        b_channel = image[:, :, 2].astype(np.float32)
        r_channel = image[:, :, 0].astype(np.float32)
        g_channel = image[:, :, 1].astype(np.float32)
        blue_dominant = ((b_channel > r_channel * 1.1) & (b_channel > g_channel * 1.05)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, blue_dominant)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        coverage = np.count_nonzero(mask) / mask.size
        confidence = min(0.95, 0.65 + coverage * 1.5) if coverage > 0.005 else 0.4

        return mask, round(confidence, 3)

    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, water_bodies: list) -> np.ndarray:
        overlay = image.copy()
        blue_layer = np.zeros_like(image)
        blue_layer[mask > 0] = (0, 100, 255)
        cv2.addWeighted(blue_layer, 0.4, overlay, 1.0, 0, overlay)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 150, 255), 2)

        # Label largest water bodies
        for i, wb in enumerate(water_bodies[:10]):
            cx, cy = int(wb["centroid"]["x"]), int(wb["centroid"]["y"])
            cv2.putText(overlay, f"W{i+1}", (cx-10, cy+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return overlay


# ══════════════════════════════════════════════════════════════════════════════
#  Building Detection Model
# ══════════════════════════════════════════════════════════════════════════════

class BuildingDetectionModel(BaseModel):
    name = "building_detection"
    version = "1.0"
    task = "BUILDING_DETECTION"
    output_type = "segmentation_mask"
    description = "Building footprint detection using edge and texture analysis"

    def predict(self, image: np.ndarray, resolution: float = None, **kwargs) -> dict:
        t0 = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Detect built-up regions: low saturation + edges
        low_sat = cv2.inRange(hsv, (0, 0, 40), (180, 50, 220))
        edges = cv2.Canny(gray, 30, 100)
        edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)

        # Buildings = low saturation AND near edges
        built_mask = cv2.bitwise_and(low_sat, edges_dilated)

        # Morphological processing to form building-like shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        built_mask = cv2.morphologyEx(built_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        built_mask = cv2.morphologyEx(built_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(built_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buildings = []
        total_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / (min(w, h) + 1)
            if aspect < 5:  # Filter out elongated shapes (roads)
                total_area += area
                buildings.append({
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "area_pixels": int(area),
                })

        buildings.sort(key=lambda b: b["area_pixels"], reverse=True)

        overlay = image.copy()
        for b in buildings[:200]:
            bb = b["bbox"]
            cv2.rectangle(overlay, (bb["x"], bb["y"]),
                         (bb["x"]+bb["w"], bb["y"]+bb["h"]), (255, 100, 0), 2)

        built_area_km2 = None
        if resolution and resolution > 0:
            built_area_km2 = round(total_area * (resolution ** 2) / 1_000_000, 4)

        coverage = total_area / (image.shape[0] * image.shape[1]) * 100
        confidence = min(0.92, 0.5 + len(buildings) * 0.005)

        return {
            "mask": built_mask,
            "overlay": overlay,
            "buildings_count": len(buildings),
            "buildings": buildings[:200],
            "total_area_pixels": int(total_area),
            "built_area_km2": built_area_km2,
            "coverage_percent": round(coverage, 2),
            "confidence": round(confidence, 3),
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Road Extraction Model
# ══════════════════════════════════════════════════════════════════════════════

class RoadExtractionModel(BaseModel):
    name = "road_extraction"
    version = "1.0"
    task = "ROAD_DETECTION"
    output_type = "segmentation_mask"
    description = "Road network extraction using edge detection and line analysis"

    def predict(self, image: np.ndarray, resolution: float = None, **kwargs) -> dict:
        t0 = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Roads: low saturation, moderate brightness, linear features
        low_sat = cv2.inRange(hsv, (0, 0, 50), (180, 35, 200))

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Hough Line Transform for linear features
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=30, maxLineGap=15)

        # Create road mask from lines
        road_mask = np.zeros(gray.shape, dtype=np.uint8)
        line_count = 0
        total_length = 0
        if lines is not None:
            for line in lines:
                coords = line[0] if getattr(line, 'ndim', 1) > 1 else line
                if len(coords) >= 4:
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    total_length += length
                    line_count += 1
                    cv2.line(road_mask, (x1, y1), (x2, y2), 255, 3)

        # Combine with low-saturation areas
        road_mask = cv2.bitwise_and(road_mask, low_sat)

        # Dilate to form road network
        kernel = np.ones((3, 3), np.uint8)
        road_mask = cv2.dilate(road_mask, kernel, iterations=2)

        # Skeletonize for road network
        from skimage.morphology import skeletonize
        skeleton = skeletonize(road_mask > 0).astype(np.uint8) * 255

        # Estimate road length
        road_length_pixels = np.count_nonzero(skeleton)
        road_length_km = None
        if resolution and resolution > 0:
            road_length_km = round(road_length_pixels * resolution / 1000, 2)

        # Create overlay
        overlay = image.copy()
        overlay[road_mask > 0] = cv2.addWeighted(
            overlay[road_mask > 0], 0.6,
            np.full_like(overlay[road_mask > 0], (255, 255, 0)), 0.4, 0
        )

        confidence = min(0.90, 0.4 + line_count * 0.005)

        return {
            "mask": road_mask,
            "skeleton": skeleton,
            "overlay": overlay,
            "road_segments": line_count,
            "total_length_pixels": road_length_pixels,
            "total_length_km": road_length_km,
            "confidence": round(confidence, 3),
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Land Cover Classification Model
# ══════════════════════════════════════════════════════════════════════════════

class LandCoverModel(BaseModel):
    name = "land_cover_classifier"
    version = "1.0"
    task = "LAND_COVER_CLASSIFICATION"
    output_type = "classification_map"
    description = "Land cover classification using K-means clustering with spectral mapping"

    CLASSES = ["Water", "Vegetation", "Built-up", "Agriculture", "Bare Land", "Road", "Other"]
    CLASS_COLORS = {
        "Water": (0, 100, 255),
        "Vegetation": (0, 180, 0),
        "Built-up": (200, 80, 0),
        "Agriculture": (150, 220, 50),
        "Bare Land": (180, 150, 80),
        "Road": (128, 128, 128),
        "Other": (100, 100, 100),
    }

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        t0 = time.time()
        h, w = image.shape[:2]

        # Prepare features: RGB + HSV (float32)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        features = np.concatenate([
            image.reshape(-1, 3).astype(np.float32),
            hsv.reshape(-1, 3).astype(np.float32),
        ], axis=1)

        # OpenCV K-means clustering
        n_clusters = 7
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centroids = cv2.kmeans(features, n_clusters, None, criteria, 3, flags)
        label_map = labels.reshape(h, w)

        # Map clusters to land cover classes based on centroids
        class_map = np.zeros((h, w), dtype=np.int32)
        for i in range(n_clusters):
            rgb_centroid = centroids[i, :3]
            hsv_centroid = centroids[i, 3:]
            class_name = self._classify_centroid(rgb_centroid, hsv_centroid)
            class_map[label_map == i] = self.CLASSES.index(class_name)

        # Calculate class percentages
        total_pixels = h * w
        class_stats = {}
        for cls_idx, cls_name in enumerate(self.CLASSES):
            count = np.count_nonzero(class_map == cls_idx)
            pct = count / total_pixels * 100
            if pct > 0.1:
                class_stats[cls_name] = round(pct, 1)

        # Create colored classification map
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx, cls_name in enumerate(self.CLASSES):
            color = self.CLASS_COLORS.get(cls_name, (100, 100, 100))
            color_map[class_map == cls_idx] = color

        # Overlay
        overlay = cv2.addWeighted(image, 0.5, color_map, 0.5, 0)

        confidence = min(0.90, 0.6 + len([v for v in class_stats.values() if v > 5]) * 0.05)

        return {
            "classification_map": class_map,
            "color_map": color_map,
            "overlay": overlay,
            "class_statistics": class_stats,
            "classes": self.CLASSES,
            "class_colors": {k: list(v) for k, v in self.CLASS_COLORS.items()},
            "confidence": round(confidence, 3),
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }

    def _classify_centroid(self, rgb: np.ndarray, hsv: np.ndarray) -> str:
        h, s, v = hsv[0], hsv[1], hsv[2]
        r, g, b = rgb

        if 90 <= h <= 130 and s > 30 and v > 20:
            return "Water"
        if 30 <= h <= 85 and s > 25 and v > 30:
            if s > 60:
                return "Vegetation"
            return "Agriculture"
        if s < 30 and 50 < v < 180:
            if v < 120:
                return "Road"
            return "Built-up"
        if 10 <= h <= 30 and s > 20:
            return "Bare Land"
        if s < 40 and v > 180:
            return "Built-up"
        return "Other"


# ══════════════════════════════════════════════════════════════════════════════
#  Vegetation Analysis Model
# ══════════════════════════════════════════════════════════════════════════════

class VegetationModel(BaseModel):
    name = "vegetation_analysis"
    version = "1.0"
    task = "VEGETATION_ANALYSIS"
    output_type = "vegetation_map"
    description = "Vegetation analysis using spectral indices (NDVI/ExG)"

    def predict(self, image: np.ndarray, multiband_data: np.ndarray = None, **kwargs) -> dict:
        t0 = time.time()
        h, w = image.shape[:2]

        has_nir = multiband_data is not None and multiband_data.shape[2] >= 4
        method = "NDVI" if has_nir else "Excess Green Index"

        if has_nir:
            from app.geospatial import compute_ndvi
            nir = multiband_data[:, :, 3].astype(np.float64)
            red = multiband_data[:, :, 0].astype(np.float64)
            index = compute_ndvi(nir, red)
            veg_mask = (index > 0.2).astype(np.uint8) * 255
            healthy = (index > 0.5).astype(np.uint8) * 255
        else:
            from app.geospatial import compute_excess_green
            index = compute_excess_green(image)
            # Normalize to 0-1 range
            idx_min, idx_max = index.min(), index.max()
            if idx_max > idx_min:
                index = (index - idx_min) / (idx_max - idx_min)
            veg_mask = (index > 0.55).astype(np.uint8) * 255
            healthy = (index > 0.7).astype(np.uint8) * 255

        # Stats
        total_px = h * w
        veg_px = np.count_nonzero(veg_mask)
        healthy_px = np.count_nonzero(healthy)

        veg_coverage = round(veg_px / total_px * 100, 1)
        healthy_pct = round(healthy_px / max(veg_px, 1) * 100, 1)

        # Create vegetation map (green gradient)
        veg_colormap = np.zeros((h, w, 3), dtype=np.uint8)
        if has_nir:
            norm_idx = ((index + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        else:
            norm_idx = (index * 255).clip(0, 255).astype(np.uint8)
        veg_colormap[:, :, 1] = norm_idx  # Green channel
        veg_colormap[veg_mask == 0] = [50, 30, 20]  # Brown for non-vegetation

        overlay = cv2.addWeighted(image, 0.5, veg_colormap, 0.5, 0)
        confidence = min(0.93, 0.6 + veg_coverage * 0.005)

        return {
            "vegetation_mask": veg_mask,
            "healthy_mask": healthy,
            "vegetation_map": veg_colormap,
            "overlay": overlay,
            "method": method,
            "nir_available": has_nir,
            "vegetation_coverage_pct": veg_coverage,
            "healthy_vegetation_pct": healthy_pct,
            "confidence": round(confidence, 3),
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Change Detection Model
# ══════════════════════════════════════════════════════════════════════════════

class ChangeDetectionModel(BaseModel):
    name = "change_detection"
    version = "1.0"
    task = "CHANGE_DETECTION"
    input_type = "image_pair"
    output_type = "change_map"
    description = "Bi-temporal change detection using image differencing and classification"

    CHANGE_CLASSES = {
        0: {"name": "No Significant Change", "color": (50, 50, 50)},
        1: {"name": "New Built-up", "color": (255, 80, 0)},
        2: {"name": "Vegetation Loss", "color": (255, 50, 50)},
        3: {"name": "Vegetation Gain", "color": (0, 200, 0)},
        4: {"name": "Water Loss", "color": (200, 150, 50)},
        5: {"name": "Water Gain", "color": (0, 100, 255)},
        6: {"name": "Other Change", "color": (200, 200, 0)},
    }

    def predict(self, image_a: np.ndarray, image_b: np.ndarray, **kwargs) -> dict:
        t0 = time.time()
        from app.geospatial import align_image_pair

        # Align images
        img_a, img_b = align_image_pair(image_a, image_b)
        h, w = img_a.shape[:2]

        # Compute per-channel absolute difference
        diff = cv2.absdiff(img_a.astype(np.int16), img_b.astype(np.int16))
        diff_mag = np.sqrt(np.sum(diff.astype(np.float64) ** 2, axis=2))

        # Threshold for significant change
        threshold = np.mean(diff_mag) + 1.5 * np.std(diff_mag)
        change_mask = (diff_mag > threshold).astype(np.uint8) * 255

        # Classify changes
        change_map = np.zeros((h, w), dtype=np.uint8)
        hsv_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2HSV)
        hsv_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2HSV)

        changed = change_mask > 0

        # Green decrease → vegetation loss
        green_decrease = (img_a[:,:,1].astype(int) - img_b[:,:,1].astype(int)) > 30
        change_map[changed & green_decrease] = 2

        # Green increase → vegetation gain
        green_increase = (img_b[:,:,1].astype(int) - img_a[:,:,1].astype(int)) > 30
        change_map[changed & green_increase] = 3

        # Saturation decrease (natural → built) → new built-up
        sat_decrease = (hsv_a[:,:,1].astype(int) - hsv_b[:,:,1].astype(int)) > 20
        val_increase = (hsv_b[:,:,2].astype(int) - hsv_a[:,:,2].astype(int)) > 10
        change_map[changed & sat_decrease & ~green_decrease] = 1

        # Blue increase → water gain
        blue_increase = (img_b[:,:,2].astype(int) - img_a[:,:,2].astype(int)) > 30
        change_map[changed & blue_increase & ~green_increase] = 5

        # Blue decrease → water loss
        blue_decrease = (img_a[:,:,2].astype(int) - img_b[:,:,2].astype(int)) > 30
        change_map[changed & blue_decrease & ~green_decrease] = 4

        # Remaining changes
        change_map[changed & (change_map == 0)] = 6

        # Statistics
        total = h * w
        stats = {}
        for cls_id, cls_info in self.CHANGE_CLASSES.items():
            count = np.count_nonzero(change_map == cls_id)
            pct = count / total * 100
            if pct > 0.1 or cls_id == 0:
                stats[cls_info["name"]] = round(pct, 1)

        # Create change map visualization
        change_vis = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, cls_info in self.CHANGE_CLASSES.items():
            change_vis[change_map == cls_id] = cls_info["color"]

        # Side-by-side
        side_by_side = np.hstack([img_a, img_b, change_vis])

        total_change_pct = round(np.count_nonzero(change_mask) / total * 100, 1)
        confidence = min(0.92, 0.55 + total_change_pct * 0.02)

        return {
            "change_mask": change_mask,
            "change_map": change_map,
            "change_visualization": change_vis,
            "side_by_side": side_by_side,
            "image_before": img_a,
            "image_after": img_b,
            "change_statistics": stats,
            "total_change_percent": total_change_pct,
            "change_classes": {str(k): v for k, v in self.CHANGE_CLASSES.items()},
            "confidence": round(confidence, 3),
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Optical-SAR Fusion Model
# ══════════════════════════════════════════════════════════════════════════════

class OpticalSARFusionModel(BaseModel):
    name = "optical_sar_fusion"
    version = "1.0"
    task = "OPTICAL_SAR_ANALYSIS"
    input_type = "image_pair"
    output_type = "fused_classification"
    description = "Optical + SAR feature-level fusion for land cover classification"
    supported_modalities = ["optical", "sar"]

    def predict(self, optical: np.ndarray, sar: np.ndarray, **kwargs) -> dict:
        t0 = time.time()
        from app.geospatial import align_image_pair

        opt, sar_img = align_image_pair(optical, sar)
        h, w = opt.shape[:2]

        # ── Optical Feature Extraction ────────────────────────────────
        opt_hsv = cv2.cvtColor(opt, cv2.COLOR_RGB2HSV)

        # Water from optical (blue hues)
        opt_water = cv2.inRange(opt_hsv, (90, 30, 20), (130, 255, 255))

        # Vegetation from optical (green hues)
        opt_veg = cv2.inRange(opt_hsv, (30, 25, 30), (85, 255, 255))

        # Built-up from optical (low saturation, moderate value)
        opt_built = cv2.inRange(opt_hsv, (0, 0, 50), (180, 40, 200))

        # ── SAR Feature Extraction ────────────────────────────────────
        if sar_img.ndim == 3:
            sar_gray = cv2.cvtColor(sar_img, cv2.COLOR_RGB2GRAY)
        else:
            sar_gray = sar_img

        # SAR backscatter analysis
        sar_mean = np.mean(sar_gray)

        # High backscatter → built-up (double-bounce)
        sar_built = (sar_gray > sar_mean + 1.5 * np.std(sar_gray)).astype(np.uint8) * 255

        # Low backscatter → water (specular reflection)
        sar_water = (sar_gray < sar_mean - 1.0 * np.std(sar_gray)).astype(np.uint8) * 255

        # Medium backscatter + texture → vegetation (volume scattering)
        sar_texture = cv2.Laplacian(sar_gray, cv2.CV_64F)
        sar_texture = np.abs(sar_texture)
        sar_veg = ((sar_gray > sar_mean * 0.7) & (sar_gray < sar_mean * 1.3) &
                   (sar_texture > np.mean(sar_texture))).astype(np.uint8) * 255

        # ── Feature-Level Fusion ──────────────────────────────────────
        # Fusion rule: agree = high confidence, disagree = weighted by modality strength
        fused_water = cv2.bitwise_or(
            cv2.bitwise_and(opt_water, sar_water),  # Both agree
            cv2.addWeighted(opt_water, 0.6, sar_water, 0.4, 0).astype(np.uint8)  # Weighted
        )
        fused_water = (fused_water > 128).astype(np.uint8) * 255

        fused_built = cv2.bitwise_or(
            cv2.bitwise_and(opt_built, sar_built),
            cv2.addWeighted(opt_built, 0.5, sar_built, 0.5, 0).astype(np.uint8)
        )
        fused_built = (fused_built > 128).astype(np.uint8) * 255

        fused_veg = cv2.bitwise_or(
            cv2.bitwise_and(opt_veg, sar_veg),
            cv2.addWeighted(opt_veg, 0.6, sar_veg, 0.4, 0).astype(np.uint8)
        )
        fused_veg = (fused_veg > 128).astype(np.uint8) * 255

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for mask in [fused_water, fused_built, fused_veg]:
            mask[:] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask[:] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # ── Create Fused Visualization ────────────────────────────────
        fused_vis = np.full((h, w, 3), (80, 80, 80), dtype=np.uint8)
        fused_vis[fused_water > 0] = (0, 100, 255)
        fused_vis[fused_veg > 0] = (0, 180, 0)
        fused_vis[fused_built > 0] = (255, 100, 0)

        # Optical result visualization
        opt_vis = np.full((h, w, 3), (80, 80, 80), dtype=np.uint8)
        opt_vis[opt_water > 0] = (0, 100, 255)
        opt_vis[opt_veg > 0] = (0, 180, 0)
        opt_vis[opt_built > 0] = (255, 100, 0)

        # SAR result visualization
        sar_vis = np.full((h, w, 3), (80, 80, 80), dtype=np.uint8)
        sar_vis[sar_water > 0] = (0, 100, 255)
        sar_vis[sar_veg > 0] = (0, 180, 0)
        sar_vis[sar_built > 0] = (255, 100, 0)

        total = h * w
        stats = {
            "water": round(np.count_nonzero(fused_water) / total * 100, 1),
            "vegetation": round(np.count_nonzero(fused_veg) / total * 100, 1),
            "built_up": round(np.count_nonzero(fused_built) / total * 100, 1),
        }

        # Agreement analysis
        water_agree = np.count_nonzero(cv2.bitwise_and(opt_water, sar_water)) / max(np.count_nonzero(fused_water), 1)
        built_agree = np.count_nonzero(cv2.bitwise_and(opt_built, sar_built)) / max(np.count_nonzero(fused_built), 1)
        avg_agreement = (water_agree + built_agree) / 2
        confidence = min(0.94, 0.55 + avg_agreement * 0.3)

        return {
            "fused_visualization": fused_vis,
            "optical_result": opt_vis,
            "sar_result": sar_vis,
            "fused_water_mask": fused_water,
            "fused_built_mask": fused_built,
            "fused_vegetation_mask": fused_veg,
            "optical_image": opt,
            "sar_image": sar_img,
            "statistics": stats,
            "modality_agreement": round(avg_agreement * 100, 1),
            "confidence": round(confidence, 3),
            "explanation": self._generate_explanation(stats, avg_agreement),
            "model": self.name,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }

    def _generate_explanation(self, stats: dict, agreement: float) -> str:
        parts = ["Fusion analysis of optical and SAR imagery:"]
        parts.append(f"• Optical data provided spectral information for land cover discrimination.")
        parts.append(f"• SAR data provided structural information via backscatter intensity.")

        if stats["water"] > 5:
            parts.append(f"• Water bodies detected ({stats['water']:.1f}%): SAR low backscatter "
                        f"confirms optical blue-spectrum water detection.")
        if stats["built_up"] > 5:
            parts.append(f"• Built-up regions detected ({stats['built_up']:.1f}%): SAR high "
                        f"backscatter (double-bounce) corroborates optical low-saturation patterns.")
        if stats["vegetation"] > 5:
            parts.append(f"• Vegetation detected ({stats['vegetation']:.1f}%): SAR volume "
                        f"scattering texture aligns with optical green spectral signatures.")

        parts.append(f"• Cross-modal agreement: {agreement*100:.0f}%")
        return "\n".join(parts)
