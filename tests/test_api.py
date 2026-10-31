"""API and pipeline regression tests for the Q-SCOPE backend."""
from __future__ import annotations

import math
from dataclasses import replace

from fastapi.testclient import TestClient

import app.api.routes as routes
from app.main import app


def valid_payload(sample_count: int = 256) -> dict:
    """Build deterministic sinusoidal telemetry matching the public schema."""
    fs = 1000.0
    frequency = 50.0
    phase = 0.02
    samples = [math.cos(2.0 * math.pi * frequency * index / fs + phase) for index in range(sample_count)]
    return {
        "shot_id": "HGV-TEST-001",
        "sensor_trace_id": "NV-CENTER-01",
        "calibration_version": "v2.1.0",
        "calibration_valid": True,
        "population_ratio_interval_s": [0.1, 0.2],
        "laser_pulse_times_s": [0.0, 0.1, 0.2],
        "samples": samples,
        "sampling_rate_hz": fs,
        "carrier_frequency_hz": frequency,
        "phase_reference_rad": 0.0,
        "phase_tolerance_rad": 0.1,
        "max_phase_uncertainty_rad": 0.15,
        "kaiser_beta": 6.0,
        "signal_coefficients": [1.0, 1.0],
        "max_gate_depth": 4,
        "target_normalized_qfi": 0.95,
        "noise_sigma": 0.01,
        "calibration_offset_rad": 0.0,
        "pulse_graph_nodes": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ],
        "pulse_graph_edges": [[0, 1], [1, 2], [2, 3]],
        "sensitivity_function": [0, 0.1, 0.2, 0.3, 0.4],
    }


def test_health_and_agents_are_available() -> None:
    with TestClient(app) as client:
        health = client.get("/healthz")
        agents = client.get("/api/agents")
    assert health.status_code == 200
    assert health.json()["pipeline_ready"] is True
    assert agents.status_code == 200
    assert len(agents.json()) == 7
    assert {item["agent_id"] for item in agents.json()} == {f"agent{i}" for i in range(1, 8)}


def test_short_waveform_is_rejected_with_422() -> None:
    payload = valid_payload(4)
    with TestClient(app) as client:
        response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422
    assert any(item["loc"][-1] == "samples" for item in response.json()["detail"])


def test_valid_waveform_runs_all_seven_agents() -> None:
    with TestClient(app) as client:
        response = client.post("/api/analyze", json=valid_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["shot_id"] == "HGV-TEST-001"
    assert body["decision"] in {"CONTINUE", "ABSTAIN", "ESCALATE_NONLINEAR"}
    assert len(body["stages"]) == 7
    assert set(body["health_certificate"]) == {
        "acquisition_integrity",
        "signal_taper_diagnostics",
        "phase_stability",
        "quantum_readiness",
    }
    assert "filtered_samples" not in body["stages"]["agent2_signal"]
    assert body["stages"]["agent5_verifier"].get("hardware_status") in {"NOT_CONNECTED", None}


def test_graph_edges_must_reference_existing_nodes() -> None:
    payload = valid_payload()
    payload["pulse_graph_edges"] = [[0, 99]]
    with TestClient(app) as client:
        response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_invalid_calibration_fails_closed() -> None:
    payload = valid_payload()
    payload["calibration_valid"] = False
    with TestClient(app) as client:
        response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "ABSTAIN"
    assert "TELEMETRY_INVALID" in body["reason_codes"]
    assert body["stages"]["agent6_hcqa"]["status"] == "skipped"


def test_bearer_token_is_enforced_when_configured(monkeypatch) -> None:
    configured = replace(routes.settings, api_access_token="test-secret", app_env="development")
    monkeypatch.setattr(routes, "settings", configured)
    with TestClient(app) as client:
        missing = client.get("/api/agents")
        invalid = client.get("/api/agents", headers={"Authorization": "Bearer wrong"})
        accepted = client.get("/api/agents", headers={"Authorization": "Bearer test-secret"})
    assert missing.status_code == 401
    assert invalid.status_code == 403
    assert accepted.status_code == 200


def test_openapi_example_meets_the_waveform_minimum() -> None:
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
    example = schema["components"]["schemas"]["AnalyzeRequest"]["examples"][0]
    assert len(example["samples"]) >= 16
    assert len(example["pulse_graph_nodes"][0]) == 8


def test_frontend_is_served_from_api_origin() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Q-SCOPE" in response.text
    assert "Run live analysis" in response.text


def test_stream_endpoint_reports_real_stage_progress_and_final_response() -> None:
    import json

    with TestClient(app) as client:
        response = client.post("/api/analyze/stream", json=valid_payload())
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    for agent in (
        "agent1_nerve", "agent2_kaiser", "agent3_kalman", "agent4_orchestrator",
        "agent6_hcqa", "agent5_verifier", "agent7_output",
    ):
        assert agent in text
    blocks = [block for block in text.split("\n\n") if block.strip()]
    complete = next(block for block in blocks if block.startswith("event: complete"))
    payload_line = next(line[6:] for line in complete.splitlines() if line.startswith("data: "))
    body = json.loads(payload_line)
    assert body["decision"] in {"CONTINUE", "ABSTAIN", "ESCALATE_NONLINEAR"}
    assert len(body["stages"]) == 7


def test_datasets_endpoints_and_telemetry_analysis() -> None:
    with TestClient(app) as client:
        # Test listing datasets
        res = client.get("/api/datasets")
        assert res.status_code == 200
        datasets = res.json()
        assert len(datasets) >= 1
        target_file = datasets[0]["filename"]
        assert "x" in datasets[0]["channels"]

        # Test single dataset detail
        detail = client.get(f"/api/datasets/{target_file}?channel=x&count=64")
        assert detail.status_code == 200
        assert len(detail.json()["samples"]) == 64

        # Test analyzing from real dataset
        req = {
            "dataset_file": target_file,
            "channel": "x",
            "sample_count": 128,
            "offset": 0,
            "carrier_frequency_hz": 25.0,
            "phase_tolerance_rad": 3.14,
            "max_phase_uncertainty_rad": 3.14,
        }
        analyze = client.post("/api/analyze/from-dataset", json=req)
        assert analyze.status_code == 200
        resp_body = analyze.json()
        assert resp_body["decision"] in {"CONTINUE", "ABSTAIN", "ESCALATE_NONLINEAR"}
        assert len(resp_body["stages"]) == 7

