# SatQuery AI — 5-Minute SIH Final Demo Presentation Script

Follow this step-by-step presentation story during your 5-minute SIH live demonstration.

---

### Step 1: Open SIH Demo Mode (0:00 - 0:45)
1. Open the frontend dashboard at `http://localhost:5173`.
2. Click **SIH Demo Mode** on the sidebar.
3. **Say:** *"Judges, SatQuery AI is an agentic vision-language assistant for remote-sensing imagery. Rather than forcing a GIS analyst to know which deep-learning model to execute, the user asks a natural-language query, and SatQuery autonomously selects and executes the appropriate specialist models."*

---

### Step 2: Demo 1 — Single-Image VQA & Scene Understanding (0:45 - 1:30)
1. Click **DEMO 1: Single-Image Remote-Sensing VQA**.
2. **Query:** *"Describe the land-cover and major structural objects visible in this satellite observation."*
3. Point out:
   - Real-time scene decomposition (Vegetation %, Built-up %, Water %).
   - Multi-level hierarchical confidence score (e.g. 91.4%).
   - Observable execution trace showing every stage with millisecond timing.

---

### Step 3: Demo 2 — Dedicated Water Body Finder (1:30 - 2:15)
1. Click **DEMO 2: Dedicated Water Body Finder**.
2. Point out:
   - Exact count of detected water bodies.
   - Connected component labeling with total surface area in km².
   - Click **Export GeoJSON** to demonstrate GIS interoperability.

---

### Step 4: Demo 4 — Bi-Temporal Change Detection (2:15 - 3:15)
1. Click **DEMO 4: Bi-Temporal Change Analysis**.
2. **Query:** *"What changed between these two dates and where did the change occur?"*
3. Point out:
   - Co-registered temporal difference map.
   - Categorical breakdown: **New Built-Up (+X%)**, **Vegetation Loss (-Y%)**, **Water Gain/Loss**.
   - No hardcoded numbers: all statistics derived directly from differential raster matrices.

---

### Step 5: Demo 5 — Optical + SAR Multimodal Fusion (3:15 - 4:00)
1. Click **DEMO 5: Optical + SAR Joint Fusion**.
2. **Query:** *"Use the optical and SAR images together to identify built-up and water-covered regions."*
3. Point out:
   - The 3-panel comparison: Optical Spectral Classification, SAR Radar Backscatter, and Fused Consensus Map.
   - Explain how SAR double-bounce confirms urban structures and penetrates clouds, while Optical discriminates spectral composition.
   - Show the Cross-Modal Agreement metric.

---

### Step 6: Demo 6 — Multi-Step Spatial Reasoning (Flagship WOW Demo) (4:00 - 4:45)
1. Click **DEMO 6: Multi-Step Spatial Reasoning**.
2. **Query:** *"Find areas where vegetation decreased and new built-up regions appeared within 500 meters of roads."*
3. Point out the Agent Execution Trace:
   - Step 1: Building extraction.
   - Step 2: Road network extraction.
   - Step 3: Bi-temporal differencing.
   - Step 4: 500-meter spatial buffering and vector intersection.
   - Highlighted target hot-spot zones flagged with priority badges.

---

### Step 7: Report Generation & Wrap Up (4:45 - 5:00)
1. Click **Export Evaluation Report**.
2. Show the generated standalone styled executive assessment report.
3. **Conclude:** *"SatQuery AI transforms complex Earth observation data into actionable, evidence-grounded intelligence through transparent, agentic AI orchestration."*
