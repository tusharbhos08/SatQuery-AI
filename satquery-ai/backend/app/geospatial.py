"""SatQuery AI — Geospatial Processing Module

Handles GeoTIFF reading, metadata extraction, preprocessing,
spectral index computation, and vectorization.
"""
import io
import struct
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  GeoTIFF Metadata Reader
# ══════════════════════════════════════════════════════════════════════════════

class GeoTIFFMetadata:
    """Extracted metadata from a GeoTIFF/TIFF file."""
    def __init__(self):
        self.width: int = 0
        self.height: int = 0
        self.bands: int = 3
        self.dtype: str = "uint8"
        self.crs: str = "unknown"
        self.bounds: Optional[dict] = None
        self.resolution: Optional[float] = None
        self.geotransform: Optional[list] = None
        self.acquisition_date: Optional[str] = None
        self.nodata: Optional[float] = None
        self.is_geotiff: bool = False
        self.file_size: int = 0
        self.warnings: list[str] = []

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "bands": self.bands,
            "dtype": self.dtype,
            "crs": self.crs,
            "bounds": self.bounds,
            "resolution": self.resolution,
            "geotransform": self.geotransform,
            "acquisition_date": self.acquisition_date,
            "nodata": self.nodata,
            "is_geotiff": self.is_geotiff,
            "file_size": self.file_size,
            "warnings": self.warnings,
        }


def read_image_metadata(filepath: str) -> GeoTIFFMetadata:
    """Read metadata from an image file (GeoTIFF, TIFF, PNG, JPG)."""
    meta = GeoTIFFMetadata()
    path = Path(filepath)
    meta.file_size = path.stat().st_size

    ext = path.suffix.lower()

    try:
        # Try tifffile first for TIFF/GeoTIFF
        if ext in ('.tif', '.tiff', '.geotiff'):
            meta = _read_tiff_metadata(filepath, meta)
        else:
            # PNG/JPG — basic metadata via PIL
            with Image.open(filepath) as img:
                meta.width, meta.height = img.size
                if img.mode == 'RGB':
                    meta.bands = 3
                elif img.mode == 'RGBA':
                    meta.bands = 4
                elif img.mode == 'L':
                    meta.bands = 1
                else:
                    meta.bands = len(img.getbands())
                meta.dtype = "uint8"
    except Exception as e:
        meta.warnings.append(f"Error reading metadata: {str(e)}")
        # Try basic PIL fallback
        try:
            with Image.open(filepath) as img:
                meta.width, meta.height = img.size
                meta.bands = len(img.getbands())
        except Exception:
            meta.warnings.append("Could not read image file")

    return meta


def _read_tiff_metadata(filepath: str, meta: GeoTIFFMetadata) -> GeoTIFFMetadata:
    """Extract metadata from TIFF/GeoTIFF files."""
    try:
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            page = tif.pages[0]
            meta.width = page.imagewidth
            meta.height = page.imagelength
            meta.dtype = str(page.dtype)

            # Band count
            shape = page.shape
            if len(shape) == 2:
                meta.bands = 1
            elif len(shape) == 3:
                meta.bands = shape[2] if shape[2] < shape[0] else shape[0]
            else:
                meta.bands = 1

            # Check for GeoTIFF tags
            for tag in page.tags.values():
                tag_name = tag.name.lower() if hasattr(tag, 'name') else ""
                # ModelTiepointTag (33922) or ModelTransformationTag (34264)
                if tag.code == 33922:  # ModelTiepointTag
                    meta.is_geotiff = True
                    data = tag.value
                    if len(data) >= 6:
                        meta.geotransform = list(data)
                elif tag.code == 34264:  # ModelTransformationTag
                    meta.is_geotiff = True
                elif tag.code == 34735:  # GeoKeyDirectoryTag
                    meta.is_geotiff = True
                    _parse_geo_keys(tag.value, meta)
                elif tag.code == 34737:  # GeoAsciiParamsTag
                    meta.is_geotiff = True
                    if isinstance(tag.value, str) and 'EPSG' in tag.value.upper():
                        meta.crs = tag.value.strip().rstrip('|')
                elif tag.code == 33550:  # ModelPixelScaleTag
                    data = tag.value
                    if len(data) >= 2:
                        meta.resolution = float(data[0])
                elif tag.code == 42112:  # GDAL_METADATA
                    if isinstance(tag.value, str):
                        _parse_gdal_metadata(tag.value, meta)

            # Calculate bounds if we have geotransform and pixel scale
            if meta.geotransform and len(meta.geotransform) >= 6:
                _compute_bounds(meta)

            if not meta.is_geotiff:
                meta.warnings.append("File is TIFF but not GeoTIFF — no geospatial metadata")

    except ImportError:
        meta.warnings.append("tifffile not available — limited metadata extraction")
        with Image.open(filepath) as img:
            meta.width, meta.height = img.size
            meta.bands = len(img.getbands())
    except Exception as e:
        meta.warnings.append(f"TIFF metadata error: {str(e)}")
        try:
            with Image.open(filepath) as img:
                meta.width, meta.height = img.size
                meta.bands = len(img.getbands())
        except Exception:
            pass

    return meta


def _parse_geo_keys(data, meta: GeoTIFFMetadata):
    """Parse GeoKeyDirectoryTag to extract CRS info."""
    try:
        if len(data) >= 4:
            num_keys = data[3]
            for i in range(num_keys):
                offset = 4 + i * 4
                if offset + 3 < len(data):
                    key_id = data[offset]
                    value = data[offset + 3]
                    # ProjectedCSTypeGeoKey = 3072
                    if key_id == 3072 and value > 0:
                        meta.crs = f"EPSG:{value}"
                    # GeographicTypeGeoKey = 2048
                    elif key_id == 2048 and value > 0 and meta.crs == "unknown":
                        meta.crs = f"EPSG:{value}"
    except Exception:
        pass


def _parse_gdal_metadata(xml_str: str, meta: GeoTIFFMetadata):
    """Parse GDAL metadata XML string."""
    try:
        if "ACQUISITION" in xml_str.upper() or "DATE" in xml_str.upper():
            import re
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', xml_str)
            if date_match:
                meta.acquisition_date = date_match.group(1)
    except Exception:
        pass


def _compute_bounds(meta: GeoTIFFMetadata):
    """Compute geographic bounds from geotransform."""
    try:
        gt = meta.geotransform
        if len(gt) >= 6:
            # Tiepoint format: I, J, K, X, Y, Z
            x_origin = gt[3]
            y_origin = gt[4]
            if meta.resolution and meta.resolution > 0:
                x_max = x_origin + meta.width * meta.resolution
                y_min = y_origin - meta.height * meta.resolution
                meta.bounds = {
                    "west": min(x_origin, x_max),
                    "east": max(x_origin, x_max),
                    "south": min(y_origin, y_min),
                    "north": max(y_origin, y_min),
                }
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Image Loading & Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def load_image(filepath: str, max_dim: int = 1024) -> np.ndarray:
    """Load an image as numpy array, optionally downscaling large images."""
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext in ('.tif', '.tiff', '.geotiff'):
        return _load_tiff(filepath, max_dim)
    else:
        return _load_standard(filepath, max_dim)


def _load_tiff(filepath: str, max_dim: int) -> np.ndarray:
    """Load TIFF/GeoTIFF, handle multi-band, normalize to uint8 for display."""
    try:
        import tifffile
        data = tifffile.imread(filepath)
    except ImportError:
        img = Image.open(filepath)
        data = np.array(img)

    # Handle various shapes
    if data.ndim == 2:
        # Single band — expand to 3-channel grayscale
        data = np.stack([data, data, data], axis=-1)
    elif data.ndim == 3:
        if data.shape[0] < data.shape[2]:
            # Bands-first (C, H, W) → (H, W, C)
            data = np.transpose(data, (1, 2, 0))
        if data.shape[2] > 3:
            # Take first 3 bands for RGB display (keep all bands in metadata)
            data = data[:, :, :3]

    # Normalize to uint8
    if data.dtype != np.uint8:
        data = _normalize_to_uint8(data)

    # Resize if too large
    data = _resize_if_needed(data, max_dim)
    return data


def _load_standard(filepath: str, max_dim: int) -> np.ndarray:
    """Load PNG/JPG."""
    img = Image.open(filepath).convert('RGB')
    data = np.array(img)
    data = _resize_if_needed(data, max_dim)
    return data


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize arbitrary dtype to uint8 [0, 255]."""
    data = data.astype(np.float64)
    # Per-channel normalization using percentile clipping
    for c in range(data.shape[2] if data.ndim == 3 else 1):
        ch = data[:, :, c] if data.ndim == 3 else data
        p2, p98 = np.percentile(ch[ch > 0] if np.any(ch > 0) else ch, [2, 98])
        if p98 > p2:
            ch_clipped = np.clip(ch, p2, p98)
            if data.ndim == 3:
                data[:, :, c] = (ch_clipped - p2) / (p98 - p2) * 255
            else:
                data = (ch_clipped - p2) / (p98 - p2) * 255
        else:
            if data.ndim == 3:
                data[:, :, c] = 0
            else:
                data = np.zeros_like(data)
    return data.astype(np.uint8)


def _resize_if_needed(data: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize image if any dimension exceeds max_dim."""
    h, w = data.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = Image.fromarray(data)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        data = np.array(img)
    return data


def load_multiband_tiff(filepath: str) -> Optional[np.ndarray]:
    """Load ALL bands of a TIFF (not just RGB) for spectral analysis."""
    try:
        import tifffile
        data = tifffile.imread(filepath)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        elif data.ndim == 3 and data.shape[0] < data.shape[2]:
            data = np.transpose(data, (1, 2, 0))
        return data
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Spectral Index Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Compute NDVI = (NIR - RED) / (NIR + RED)."""
    nir = nir_band.astype(np.float64)
    red = red_band.astype(np.float64)
    denominator = nir + red
    ndvi = np.where(denominator > 0, (nir - red) / denominator, 0.0)
    return np.clip(ndvi, -1.0, 1.0)


def compute_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """Compute NDWI = (GREEN - NIR) / (GREEN + NIR)."""
    green = green_band.astype(np.float64)
    nir = nir_band.astype(np.float64)
    denominator = green + nir
    ndwi = np.where(denominator > 0, (green - nir) / denominator, 0.0)
    return np.clip(ndwi, -1.0, 1.0)


def compute_excess_green(image: np.ndarray) -> np.ndarray:
    """Compute Excess Green Index for vegetation from RGB: ExG = 2G - R - B."""
    img = image.astype(np.float64)
    if img.ndim == 3 and img.shape[2] >= 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        exg = 2.0 * g - r - b
        return exg
    return np.zeros(image.shape[:2])


# ══════════════════════════════════════════════════════════════════════════════
#  Pair Alignment & Compatibility
# ══════════════════════════════════════════════════════════════════════════════

def check_pair_compatibility(meta_a: GeoTIFFMetadata, meta_b: GeoTIFFMetadata) -> dict:
    """Check if two images are compatible as a pair (bi-temporal or cross-modal)."""
    result = {
        "compatible": True,
        "same_crs": True,
        "similar_dimensions": True,
        "geographic_overlap": True,
        "warnings": [],
    }

    # CRS check
    if meta_a.crs != "unknown" and meta_b.crs != "unknown":
        if meta_a.crs != meta_b.crs:
            result["same_crs"] = False
            result["warnings"].append(
                f"CRS mismatch: {meta_a.crs} vs {meta_b.crs}. Reprojection may be needed."
            )

    # Dimension check (allow 20% difference)
    dim_ratio_w = min(meta_a.width, meta_b.width) / max(meta_a.width, meta_b.width) if max(meta_a.width, meta_b.width) > 0 else 1
    dim_ratio_h = min(meta_a.height, meta_b.height) / max(meta_a.height, meta_b.height) if max(meta_a.height, meta_b.height) > 0 else 1
    if dim_ratio_w < 0.5 or dim_ratio_h < 0.5:
        result["similar_dimensions"] = False
        result["warnings"].append(
            f"Large dimension difference: {meta_a.width}x{meta_a.height} vs {meta_b.width}x{meta_b.height}"
        )

    # Geographic overlap (if bounds available)
    if meta_a.bounds and meta_b.bounds:
        overlap = _check_overlap(meta_a.bounds, meta_b.bounds)
        if not overlap:
            result["geographic_overlap"] = False
            result["compatible"] = False
            result["warnings"].append("No geographic overlap detected between images")

    if not result["same_crs"] and meta_a.crs != "unknown":
        result["warnings"].append("Images may still be compatible — manual verification recommended")

    return result


def _check_overlap(bounds_a: dict, bounds_b: dict) -> bool:
    """Check if two bounding boxes overlap."""
    try:
        return not (
            bounds_a["east"] < bounds_b["west"] or
            bounds_a["west"] > bounds_b["east"] or
            bounds_a["north"] < bounds_b["south"] or
            bounds_a["south"] > bounds_b["north"]
        )
    except (KeyError, TypeError):
        return True  # Assume overlap if bounds are incomplete


def align_image_pair(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize image pair to matching dimensions."""
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])

    a = Image.fromarray(img_a).resize((w, h), Image.LANCZOS)
    b = Image.fromarray(img_b).resize((w, h), Image.LANCZOS)

    return np.array(a), np.array(b)


# ══════════════════════════════════════════════════════════════════════════════
#  Vectorization (Mask → GeoJSON-like polygons)
# ══════════════════════════════════════════════════════════════════════════════

def mask_to_polygons(mask: np.ndarray, min_area: int = 100) -> list[dict]:
    """Convert a binary mask to polygon features (pixel coordinates)."""
    import cv2
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            coords = approx.squeeze().tolist()
            if len(coords) >= 3:
                bbox = cv2.boundingRect(cnt)
                polygons.append({
                    "coordinates": coords,
                    "area_pixels": float(area),
                    "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
                })
    return polygons


def estimate_area_km2(area_pixels: float, resolution: Optional[float]) -> Optional[float]:
    """Convert pixel area to km² using resolution (meters/pixel)."""
    if resolution and resolution > 0:
        area_m2 = area_pixels * (resolution ** 2)
        return round(area_m2 / 1_000_000, 4)
    return None
