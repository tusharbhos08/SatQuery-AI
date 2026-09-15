# SatQuery AI — “Ask Questions. Understand Earth.”

### SIH26167: Agentic Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis

SatQuery AI is an autonomous, agentic multimodal vision-language intelligence platform designed specifically for Earth observation imagery. It accepts natural-language user queries and dynamically selects, configures, and executes appropriate specialized remote-sensing AI models and GIS pipelines.

---

## 🌟 Key Capabilities & Demonstration Workflows

1. **Single-Image Remote-Sensing VQA & Captioning** — Land-cover classification, scene description, structural analysis.
2. **Text-Guided Region Grounding** — Prompt-driven localization with bounding boxes, segmentation masks, and pixel overlays.
3. **Dedicated Water Body Finder** — Connected components analysis, morphological filtering, surface area estimation in km², and GeoJSON vector polygon export.
4. **Building & Road Network Extraction** — Footprint segmentation, density calculation, Hough line transform, and skeletonized transport connectivity.
5. **Vegetation & Spectral Canopy Health** — Multi-spectral NDVI calculation with NIR bands, or calibrated Excess Green Index (ExG) fallback.
6. **Bi-Temporal Change Understanding** — Differential spectral analysis, categorical transition classification (new built-up, vegetation loss/gain, water shifts), and side-by-side change maps.
7. **Cross-Modal Optical + SAR Fusion** — Dual-branch feature extraction combining optical spectral signatures with cloud-penetrating SAR radar backscatter.
8. **Multi-Step Spatial Reasoning (Flagship WOW Demo)** — Multi-tool orchestration with spatial buffering (e.g. 500m road corridors) and vector intersection.
9. **Observable Execution Trace & Hierarchical Confidence** — Transparent audit trail from input validation to model execution without fake hardcoding.
10. **Automated Downloadable Reports** — Export evaluation assessments as standalone styled HTML/PDF documents.

---

## 🚀 Quick Start Instructions

### Prerequisites
- Python 3.10+ (Tested with Python 3.14)
- Node.js 18+ (Tested with Node.js v24.15)
- npm 9+

### Option A: One-Command Startup (Windows)

**Using PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_satquery.ps1
```

**Using Command Prompt:**
```cmd
run_satquery.bat
```

### Option B: Manual Startup

**1. Start the FastAPI Backend:**
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs & Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/`

**2. Start the React/Vite Frontend:**
```bash
cd frontend
npm run dev
```
- Dashboard UI: `http://localhost:5173`

---

## 📊 SIH Mandatory Requirement Coverage Matrix

| # | SIH Mandatory Requirement | Implementation in SatQuery AI | Status |
|---|---------------------------|--------------------------------|--------|
| 1 | Single-image remote-sensing VQA | `VQAModel` + Scene spectral decomposition | ✅ Complete |
| 2 | Text-guided region grounding / captioning | `GroundingModel` + `CaptionModel` | ✅ Complete |
| 3 | Bi-temporal change understanding | `ChangeDetectionModel` + Categorical transition map | ✅ Complete |
| 4 | Cross-modal optical/multispectral + SAR | `OpticalSARFusionModel` with agreement score | ✅ Complete |
| 5 | Agentic model/tool orchestration | `InputValidator` → `QueryUnderstanding` → `TaskOrchestrator` | ✅ Complete |
| 6 | Remote-sensing adaptation (BigEarthNet) | `ml/training/train.py` LoRA adaptation pipeline | ✅ Complete |
| 7 | GeoTIFF/TIFF support | `geospatial.py` with rasterio & tifffile metadata parser | ✅ Complete |
| 8 | Input validation & compatibility checking | `InputValidationAgent` for single & paired GeoTIFFs | ✅ Complete |
| 9 | Visual evidence generation | Real-time raster overlay generation & base64/URL rendering | ✅ Complete |
| 10 | Confidence estimation | Hierarchical 4-tier confidence breakdown | ✅ Complete |
| 11 | Observable execution trace | Step-by-step audit trace with execution millisecond timings | ✅ Complete |
| 12 | Downloadable analysis reports | Standalone HTML & printable executive assessment reports | ✅ Complete |
| 13 | Polished SIH Web Dashboard | Dark space geospatial intelligence workstation | ✅ Complete |

---

## 🏗️ Project Architecture

```
satquery-ai/
├── frontend/                     # React + TypeScript + Vite + Tailwind Dashboard
│   ├── src/
│   │   ├── components/           # Analyze, Change, OpticalSAR, WaterFinder, DemoMode, Models
│   │   ├── api.ts                # REST API client
│   │   └── types.ts              # TypeScript interfaces
├── backend/                      # Python FastAPI Backend
│   ├── app/
│   │   ├── agents.py             # Validation, Query Understanding, Orchestrator Agents
│   │   ├── ml_models.py          # 10 Remote-Sensing ML Model Implementations
│   │   ├── pipelines.py          # Execution Pipelines & Multi-Step Spatial Reasoning
│   │   ├── geospatial.py         # GeoTIFF metadata, spectral indices, vectorization
│   │   ├── services.py           # Evidence, Confidence, Reports, Synthetic Generator
│   │   ├── routes.py             # REST API routes
│   │   ├── database.py           # SQLAlchemy database schema
│   │   └── main.py               # FastAPI application entrypoint
│   └── tests/
│       └── test_pipeline.py      # Full 13-capability automated test suite
├── ml/                           # Model Adaptation & Training
│   └── training/
│       └── train.py              # BigEarthNet LoRA fine-tuning script
├── data/demo/                    # Preloaded Synthetic & Sample GeoTIFF Imagery
├── reports/                      # Generated HTML/PDF assessment reports
├── docs/                         # Comprehensive documentation suite
├── run_satquery.ps1              # One-click launch script (PowerShell)
├── run_satquery.bat              # One-click launch script (CMD)
└── docker-compose.yml            # Docker container deployment
```

---

## 🔒 Scientific Authenticity Guarantee

SatQuery AI strictly adheres to the **No Fake AI Rule**. All outputs, bounding geometries, connected component areas, change statistics, and fusion consensus maps are calculated directly from pixel data using real computer vision and spatial algorithms.
