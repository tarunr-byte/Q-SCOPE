"""SQLAlchemy models for Q-SCOPE projects and telemetry analysis history."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from app.core.database import Base


class ProjectModel(Base):
    """User flight projects with telemetry connections and runtime sessions."""
    __tablename__ = "projects"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    sensor = Column(String(255), nullable=True, default="ADXL345 Tri-Axial Accelerometer (Real Telemetry)")
    device = Column(String(255), nullable=True, default="Sensor Gateway / Serial Bridge")
    sensor_connected = Column(Boolean, default=True)
    gps = Column(String(255), nullable=True, default="USB / Serial GPS receiver")
    gps_connected = Column(Boolean, default=True)
    backend = Column(String(255), default="Aer Simulator")
    dataset = Column(String(255), default="qscope_clean_synthetic_adxl345.csv")
    created = Column(String(64), nullable=True)
    runtimes_json = Column(Text, nullable=True, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize project matching frontend data structure."""
        runtimes = []
        if self.runtimes_json:
            try:
                runtimes = json.loads(self.runtimes_json)
            except Exception:
                runtimes = []
        return {
            "id": self.id,
            "name": self.name,
            "sensor": self.sensor,
            "device": self.device,
            "sensorConnected": self.sensor_connected,
            "gps": self.gps,
            "gpsConnected": self.gps_connected,
            "backend": self.backend,
            "dataset": self.dataset,
            "created": self.created or datetime.utcnow().strftime("%d %b %Y"),
            "runtimes": runtimes,
        }


class AnalysisRunModel(Base):
    """Historical record of executed 7-agent quantum telemetry runs."""
    __tablename__ = "analysis_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, index=True, nullable=False)
    project_id = Column(String(64), index=True, nullable=True)
    runtime_id = Column(String(64), nullable=True)
    shot_id = Column(String(64), nullable=True)
    decision = Column(String(64), nullable=True)
    dataset_file = Column(String(255), nullable=True)
    channel = Column(String(16), nullable=True)
    reason_codes_json = Column(Text, nullable=True)
    stages_json = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize history record."""
        reasons = []
        stages = {}
        summary = {}
        if self.reason_codes_json:
            try:
                reasons = json.loads(self.reason_codes_json)
            except Exception:
                reasons = []
        if self.stages_json:
            try:
                stages = json.loads(self.stages_json)
            except Exception:
                stages = {}
        if self.summary_json:
            try:
                summary = json.loads(self.summary_json)
            except Exception:
                summary = {}
        return {
            "id": self.id,
            "run_id": self.run_id,
            "project_id": self.project_id,
            "runtime_id": self.runtime_id,
            "shot_id": self.shot_id,
            "decision": self.decision,
            "dataset_file": self.dataset_file,
            "channel": self.channel,
            "reason_codes": reasons,
            "stages": stages,
            "summary": summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
