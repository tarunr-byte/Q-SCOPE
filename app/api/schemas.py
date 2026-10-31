"""Validated request and response contracts for the Q-SCOPE HTTP API.

The schema rejects malformed telemetry before it can enter numerical code.
Physics-specific checks that depend on several fields are handled by a model
validator; runtime checks are repeated by Agent 1 as a fail-closed boundary.
"""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import MAX_WAVEFORM_SAMPLES, MAX_PHASE_UNCERTAINTY_RAD

Decision = Literal["CONTINUE", "ABSTAIN", "ESCALATE_NONLINEAR"]

_EXAMPLE_SAMPLES = [
    round(math.cos(2.0 * math.pi * 50.0 * index / 1000.0), 6)
    for index in range(64)
]
_EXAMPLE_REQUEST = {
    "shot_id": "HGV-TEST-001",
    "sensor_trace_id": "NV-CENTER-01",
    "calibration_version": "v2.1.0",
    "calibration_valid": True,
    "population_ratio_interval_s": [0.1, 0.2],
    "laser_pulse_times_s": [0.0, 0.1, 0.2],
    "samples": _EXAMPLE_SAMPLES,
    "sampling_rate_hz": 1000.0,
    "carrier_frequency_hz": 50.0,
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


class AnalyzeRequest(BaseModel):
    """Input telemetry for one analysis shot."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={"examples": [_EXAMPLE_REQUEST]},
    )

    shot_id: str = Field(..., min_length=1, max_length=128, description="Unique measurement ID.")
    sensor_trace_id: str = Field(..., min_length=1, max_length=128, description="Sensor trace ID.")
    calibration_version: str = Field(..., min_length=1, max_length=64, description="Calibration version.")
    calibration_valid: bool = Field(..., description="Whether calibration has been approved.")

    population_ratio_interval_s: tuple[float, float] = Field(
        ..., description="Increasing measurement interval in seconds."
    )
    laser_pulse_times_s: tuple[float, float, float] = Field(
        ..., description="Three strictly increasing pulse times in seconds."
    )

    samples: list[float] = Field(
        ...,
        min_length=16,
        max_length=MAX_WAVEFORM_SAMPLES,
        description="Finite, one-dimensional ADC samples; at least 16 are required.",
    )
    sampling_rate_hz: float = Field(1000.0, gt=0, description="ADC sampling rate in Hz.")
    carrier_frequency_hz: float = Field(50.0, gt=0, description="Demodulation frequency in Hz.")
    phase_reference_rad: float = Field(0.0, description="Expected phase in radians.")
    phase_tolerance_rad: float = Field(0.10, gt=0, le=math.pi, description="Maximum phase error.")
    max_phase_uncertainty_rad: float = Field(
        MAX_PHASE_UNCERTAINTY_RAD, gt=0, le=math.pi, description="Maximum phase uncertainty."
    )
    kaiser_beta: float = Field(6.0, ge=0, le=20, description="Kaiser-window beta parameter.")
    signal_coefficients: tuple[float, float] = Field(
        ..., description="Two coefficients for the demonstrator's quantum generator."
    )

    max_gate_depth: int = Field(4, ge=1, le=8, description="Maximum HCQA search depth.")
    target_normalized_qfi: float = Field(
        0.95, gt=0, le=1, description="Minimum normalized QFI required for circuit acceptance."
    )
    noise_sigma: float = Field(0.01, ge=0, description="Declared environmental noise level.")
    calibration_offset_rad: float = Field(0.0, description="Additive phase calibration offset.")

    pulse_graph_nodes: list[list[float]] = Field(
        ..., min_length=1, max_length=10000, description="Graph nodes with 8 numeric features each."
    )
    pulse_graph_edges: list[tuple[int, int]] = Field(
        ..., max_length=50000, description="Directed edge endpoints as zero-based node indices."
    )
    sensitivity_function: list[float] = Field(
        ..., min_length=2, max_length=10000, description="Finite sensitivity samples."
    )

    @field_validator("shot_id", "sensor_trace_id", "calibration_version")
    @classmethod
    def validate_nonblank_metadata(cls, value: str) -> str:
        """Reject empty identifiers, including values made empty by trimming."""
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator(
        "samples", "sensitivity_function", "signal_coefficients", "population_ratio_interval_s",
        "laser_pulse_times_s", "pulse_graph_nodes", "phase_reference_rad", "phase_tolerance_rad",
        "max_phase_uncertainty_rad", "kaiser_beta", "target_normalized_qfi", "noise_sigma",
        "calibration_offset_rad", "sampling_rate_hz", "carrier_frequency_hz",
    )
    @classmethod
    def validate_finite_numeric_values(cls, value: Any) -> Any:
        """Reject NaN and infinity so numerical operations remain predictable."""
        def finite(item: Any) -> bool:
            if isinstance(item, (int, float)):
                return math.isfinite(float(item))
            if isinstance(item, (list, tuple)):
                return all(finite(child) for child in item)
            return True

        if not finite(value):
            raise ValueError("all numeric values must be finite")
        return value

    @field_validator("pulse_graph_nodes")
    @classmethod
    def validate_node_feature_width(cls, nodes: list[list[float]]) -> list[list[float]]:
        """Require the fixed feature width consumed by graph diagnostics."""
        for index, node in enumerate(nodes):
            if len(node) != 8:
                raise ValueError(f"pulse_graph_nodes[{index}] must contain exactly 8 features")
        return nodes

    @field_validator("pulse_graph_edges")
    @classmethod
    def validate_edge_pairs(cls, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Reject malformed/self-loop endpoints at the schema boundary."""
        for index, (source, target) in enumerate(edges):
            if source < 0 or target < 0:
                raise ValueError(f"pulse_graph_edges[{index}] endpoints must be non-negative")
            if source == target:
                raise ValueError(f"pulse_graph_edges[{index}] must not be a self-loop")
        return edges

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "AnalyzeRequest":
        """Check ordering, graph endpoint bounds and the sampling Nyquist limit."""
        start, end = self.population_ratio_interval_s
        if start >= end:
            raise ValueError("population_ratio_interval_s must be strictly increasing")
        p0, p1, p2 = self.laser_pulse_times_s
        if not (p0 < p1 < p2):
            raise ValueError("laser_pulse_times_s must be strictly increasing")
        if self.carrier_frequency_hz >= self.sampling_rate_hz / 2:
            raise ValueError("carrier_frequency_hz must be below the Nyquist frequency")
        if not any(abs(value) > 1e-12 for value in self.signal_coefficients):
            raise ValueError("at least one signal coefficient must be non-zero")
        node_count = len(self.pulse_graph_nodes)
        for index, (source, target) in enumerate(self.pulse_graph_edges):
            if source >= node_count or target >= node_count:
                raise ValueError(
                    f"pulse_graph_edges[{index}] references a node outside 0..{node_count - 1}"
                )
        return self


class AnalyzeResponse(BaseModel):
    """Stable result contract returned after all reachable agents have run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="Unique ID for this invocation.")
    shot_id: str = Field(..., description="Echo of the request's shot ID.")
    decision: Decision = Field(..., description="Fail-closed orchestrator decision.")
    reason_codes: list[str] = Field(..., description="Deterministic decision reason codes.")
    stages: dict[str, dict[str, Any]] = Field(..., description="Public outputs for all seven agents.")
    health_certificate: dict[str, Any] = Field(..., description="Four-dimensional software health certificate.")
    llm_notes: dict[str, str] = Field(default_factory=dict, description="Optional advisory summaries.")
    capability_limits: list[str] = Field(default_factory=list, description="Limits or fallbacks used.")
    model_scope: str = Field(
        "software diagnostics only; no physical hardware certification",
        description="Scope statement preventing over-interpretation of simulator output.",
    )


class AgentStatus(BaseModel):
    """Health and capability metadata for a single pipeline agent."""

    agent_id: str
    name: str
    role: str
    status: Literal["ready", "degraded"]
    model_configured: bool
    llm_optional: bool = True
    description: str


class HealthResponse(BaseModel):
    """Lightweight liveness/readiness response."""

    status: Literal["ok", "degraded"]
    service: str = "q-scope"
    workflow: str = "seven-agent-pipeline"
    pipeline_ready: bool


class DatasetSummary(BaseModel):
    """Metadata describing a sensor telemetry CSV dataset."""

    filename: str
    name: str
    row_count: int
    channels: list[str]
    sample_rate_hz: float
    preview_samples: dict[str, list[float]]


class DatasetAnalyzeRequest(BaseModel):
    """Request to slice a real CSV sensor dataset and execute the 7-agent pipeline."""

    dataset_file: str = Field(..., description="Filename of CSV in the dataset directory")
    channel: Literal["x", "y", "z", "magnitude"] = Field("x", description="Sensor channel/axis")
    sample_count: int = Field(128, ge=16, le=4096, description="Number of samples to analyze")
    offset: int = Field(0, ge=0, description="Offset index into the dataset")
    shot_id: str = Field("DATASET-SHOT-001", description="Shot identifier")
    sensor_trace_id: str = Field("ADXL345-IMU", description="Sensor trace identifier")
    carrier_frequency_hz: float = Field(25.0, gt=0, description="Demodulation / carrier frequency in Hz")
    phase_tolerance_rad: float = Field(0.5, gt=0, le=math.pi, description="Maximum allowable phase error in radians")
    max_phase_uncertainty_rad: float = Field(0.5, gt=0, le=math.pi, description="Maximum allowable phase uncertainty in radians")
    kaiser_beta: float = Field(6.0, ge=0, le=20.0, description="Kaiser window parameter")
    target_normalized_qfi: float = Field(0.95, gt=0, le=1.0, description="Target QFI")
    max_gate_depth: int = Field(4, ge=1, le=8, description="Max circuit gate depth")


# =====================================================================
# AZHVA AI SCHEMAS (Phase 1 Translation & Phase 2 Copilot Chat)
# =====================================================================

class AzhvaTranslateRequest(BaseModel):
    """Input for Azhva Phase 1 (Machine-to-Human Translation)."""
    model_config = ConfigDict(extra="forbid")
    samples: list[float] = Field(..., min_length=16, description="Raw ADC samples.")
    sampling_rate_hz: float = Field(200.0, gt=0, description="ADC sampling rate in Hz.")


class AzhvaTranslateResponse(BaseModel):
    """Output for Azhva Phase 1."""
    model_config = ConfigDict(extra="forbid")
    machine_voice: str = Field(..., description="The machine's translated message to the human.")
    perception_notes: str = Field(..., description="Raw visual diagnosis from the model.")
    spectrogram_base64: str = Field(..., description="Base64 encoded spectrogram image.")


class AzhvaChatRequest(BaseModel):
    """Input for Azhva Phase 2 (App Copilot)."""
    model_config = ConfigDict(extra="forbid")
    message: str = Field(..., min_length=1, description="User's question.")
    session_id: str = Field(..., description="Session identifier for context tracking.")


class AzhvaChatResponse(BaseModel):
    """Output for Azhva Phase 2."""
    model_config = ConfigDict(extra="forbid")
    reply: str = Field(..., description="The Copilot's response.")




# =====================================================================
# PROJECT & RUNTIME DATABASE SCHEMAS
# =====================================================================

class ProjectCreateRequest(BaseModel):
    """Payload to create or update a user project."""
    id: str
    name: str
    sensor: str | None = "ADXL345 Tri-Axial Accelerometer (Real Telemetry)"
    device: str | None = "Sensor Gateway / Serial Bridge"
    sensorConnected: bool | None = True
    gps: str | None = "USB / Serial GPS receiver"
    gpsConnected: bool | None = True
    backend: str | None = "Aer Simulator"
    dataset: str | None = "qscope_clean_synthetic_adxl345.csv"
    created: str | None = None
    runtimes: list[dict[str, Any]] | None = []
