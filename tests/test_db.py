"""Tests for Q-SCOPE database persistence and project management."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_database_projects_crud():
    """Verify projects can be listed, created, and retrieved from database."""
    # 1. List projects
    res = client.get("/api/projects")
    assert res.status_code == 200
    initial_count = len(res.json())

    # 2. Create a fresh project
    test_id = "test_flight_001"
    payload = {
        "id": test_id,
        "name": "Live Orbital Insertion Test",
        "sensor": "ADXL345 Tri-Axial Accelerometer (Real Telemetry)",
        "device": "IoT edge gateway (ADXL345)",
        "sensorConnected": True,
        "gps": "USB / Serial GPS receiver",
        "gpsConnected": True,
        "backend": "Aer Simulator",
        "dataset": "qscope_clean_synthetic_adxl345.csv",
        "created": "31 Oct 2026",
        "runtimes": [
            {
                "id": "r_test_01",
                "number": "01",
                "name": "Live Boundary Layer Verification",
                "status": "Stopped",
                "timestamp": "Not yet started",
                "dataset_file": "qscope_clean_synthetic_adxl345.csv",
                "channel": "x",
                "sample_count": 256,
            }
        ],
    }
    create_res = client.post("/api/projects", json=payload)
    assert create_res.status_code == 200
    data = create_res.json()
    assert data["id"] == test_id
    assert data["name"] == "Live Orbital Insertion Test"
    assert len(data["runtimes"]) == 1

    # 3. Verify it is listed in DB
    list_res = client.get("/api/projects")
    assert list_res.status_code == 200
    items = list_res.json()
    assert len(items) == initial_count + 1
    found = [p for p in items if p["id"] == test_id]
    assert len(found) == 1

    # 4. Clean up / Delete
    del_res = client.delete(f"/api/projects/{test_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_id"] == test_id

    # 5. Verify it is removed
    final_res = client.get("/api/projects")
    assert final_res.status_code == 200
    assert len(final_res.json()) == initial_count
