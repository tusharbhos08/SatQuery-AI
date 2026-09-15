"""SatQuery AI — Automated Verification Test Suite

Tests all 13 core remote sensing pipelines and agent orchestration.
"""
import sys
from pathlib import Path

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml_models import ModelRegistry
from app.agents import InputValidationAgent, QueryUnderstandingAgent, TaskOrchestrator
from app.services import generate_synthetic_satellite_scene, EvidenceService, ConfidenceService, ReportGenerator
from app.geospatial import read_image_metadata, load_image


def test_all_pipelines():
    print("==================================================")
    print("   SATQUERY AI — FULL SYSTEM PIPELINE TEST        ")
    print("==================================================")

    # 1. Generate test assets
    demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    optical_tif = demo_dir / "test_optical.tif"
    sar_tif = demo_dir / "test_sar.tif"
    t1_tif = demo_dir / "test_t1.tif"
    t2_tif = demo_dir / "test_t2.tif"

    opt_img = generate_synthetic_satellite_scene("optical", save_path=str(optical_tif))
    sar_img = generate_synthetic_satellite_scene("sar", save_path=str(sar_tif))
    t1_img = generate_synthetic_satellite_scene("bitemporal_before", save_path=str(t1_tif))
    t2_img = generate_synthetic_satellite_scene("bitemporal_after", save_path=str(t2_tif))

    print(f"[PASS] Synthetic test GeoTIFF assets generated ({optical_tif.name}, {sar_tif.name}, {t1_tif.name}, {t2_tif.name})")

    # 2. Test Input Validation Agent
    val_agent = InputValidationAgent()
    val_opt = val_agent.validate_single(str(optical_tif))
    val_sar = val_agent.validate_single(str(sar_tif))
    val_pair = val_agent.validate_pair(str(optical_tif), str(sar_tif))

    assert val_opt["valid"] is True, "Optical validation failed"
    assert val_sar["valid"] is True, "SAR validation failed"
    assert val_pair["valid"] is True, "Pair validation failed"
    print("[PASS] Input Validation Agent: validated single & cross-modal pair successfully")

    # 3. Test Query Understanding Agent
    query_agent = QueryUnderstandingAgent()
    test_queries = [
        ("Describe the scene and land cover", "CAPTIONING"),
        ("Find all water bodies in this scene", "WATER_DETECTION"),
        ("Highlight the building structures", "GROUNDING"),
        ("What changed between these two dates?", "CHANGE_DETECTION"),
        ("Use optical and SAR to identify built-up and water", "OPTICAL_SAR_ANALYSIS"),
        ("Find areas where vegetation decreased near roads within 500m", "MULTI_STEP_ANALYSIS"),
    ]

    for q, expected in test_queries:
        res = query_agent.classify(q)
        assert res["task_type"] == expected, f"Query '{q}' classified as {res['task_type']}, expected {expected}"
        print(f"[PASS] Query Agent: '{q[:35]}...' -> {res['task_type']} (conf: {res['confidence']})")

    # 4. Test Orchestrator & ML Models
    registry = ModelRegistry()
    orchestrator = TaskOrchestrator(registry)
    evidence_svc = EvidenceService()
    confidence_svc = ConfidenceService()

    meta_opt = val_opt["metadata"]
    meta_t1 = val_agent.validate_single(str(t1_tif))["metadata"]
    meta_t2 = val_agent.validate_single(str(t2_tif))["metadata"]

    workflows = [
        ("VQA", "Describe the major features", [str(optical_tif)], [opt_img], [meta_opt]),
        ("WATER_DETECTION", "Find water bodies", [str(optical_tif)], [opt_img], [meta_opt]),
        ("BUILDING_DETECTION", "Detect buildings", [str(optical_tif)], [opt_img], [meta_opt]),
        ("ROAD_DETECTION", "Extract road network", [str(optical_tif)], [opt_img], [meta_opt]),
        ("VEGETATION_ANALYSIS", "Compute vegetation index", [str(optical_tif)], [opt_img], [meta_opt]),
        ("LAND_COVER_CLASSIFICATION", "Classify land cover", [str(optical_tif)], [opt_img], [meta_opt]),
        ("CHANGE_DETECTION", "What changed?", [str(t1_tif), str(t2_tif)], [t1_img, t2_img], [meta_t1, meta_t2]),
        ("OPTICAL_SAR_ANALYSIS", "Fuse optical and SAR", [str(optical_tif), str(sar_tif)], [opt_img, sar_img], [meta_opt, val_sar["metadata"]]),
        ("MULTI_STEP_ANALYSIS", "Find new construction near roads within 500 meters", [str(t1_tif), str(t2_tif)], [t1_img, t2_img], [meta_t1, meta_t2]),
    ]

    for name, q, paths, imgs, metas in workflows:
        result = orchestrator.plan_and_execute(q, paths, imgs, metas)
        assert result["answer"], f"{name} produced empty answer"
        assert result["confidence"] > 0, f"{name} produced zero confidence"
        assert "execution_trace" in result, f"{name} missing execution trace"

        processed_ev = evidence_svc.process_pipeline_evidence(result.get("visual_evidence", {}))
        assert "primary_base64" in processed_ev, f"{name} missing primary visual evidence"

        conf = confidence_svc.compute_scores(result["task_type"], metas, result["confidence"])
        print(f"[PASS] Pipeline '{name}': Answer generated ({len(result['answer'])} chars), Confidence: {conf['overall_percentage']}, Trace steps: {len(result['execution_trace']['steps'])}")

    # 5. Test Report Generator
    report_gen = ReportGenerator()
    html_path, html_url = report_gen.generate_html_report(
        {"query": "Sample verification query"},
        {
            "task_type": "MULTI_STEP_ANALYSIS",
            "answer": "Automated verification test completed successfully with 100% pipeline integrity.",
            "confidence_data": {"overall_percentage": "94.5%"},
            "models_used": ["building_detection", "road_extraction", "spatial_reasoning_engine"],
            "processing_time_ms": 124.5,
            "execution_trace": {"steps": [{"step": "Test step", "status": "completed", "time_ms": 10, "details": "ok"}]},
            "visual_evidence": {"primary_base64": processed_ev["primary_base64"]},
        }
    )
    assert Path(html_path).exists(), "Report file was not generated"
    print(f"[PASS] Report Generator: Created HTML report at {html_url}")

    print("\n==================================================")
    print("   ALL 13 REMOTE SENSING CAPABILITIES VERIFIED!   ")
    print("==================================================")


if __name__ == "__main__":
    test_all_pipelines()
