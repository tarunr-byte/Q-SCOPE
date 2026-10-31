"""HTTP routes for Q-SCOPE analysis, live pipeline progress, and readiness."""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
from pathlib import Path
import secrets
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from agents.azhva_agent import run_phase_1_translation, run_phase_2_chat
from agents.graph import qscope_engine
from agents.models import model_configured
from agents.registry import AGENT_REGISTRY
from agents.state import initialize_state
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.models import ProjectModel, AnalysisRunModel
from app.api.schemas import (
    AgentStatus,
    AnalyzeRequest,
    AnalyzeResponse,
    AzhvaChatRequest,
    AzhvaChatResponse,
    AzhvaTranslateRequest,
    AzhvaTranslateResponse,
    DatasetAnalyzeRequest,
    DatasetSummary,
    HealthResponse,
    ProjectCreateRequest,
)
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


async def verify_token(authorization: str | None = Header(default=None)) -> bool:
    """Validate optional bearer auth; production mode cannot silently bypass it."""
    expected_token = settings.api_access_token
    if not expected_token:
        if settings.auth_required:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API authentication is required but API_ACCESS_TOKEN is not configured.",
            )
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A Bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    supplied_token = authorization[len("Bearer "):].strip()
    if not supplied_token or not secrets.compare_digest(supplied_token, expected_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API access token.")
    return True


def _public_signal(signal: dict[str, Any]) -> dict[str, Any]:
    """Remove bulky waveform/window vectors from the JSON response payload."""
    hidden_fields = {"filtered_samples", "window", "detail"}
    return {key: value for key, value in signal.items() if key not in hidden_fields}


def _public_stages(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Expose a compact record of each agent without leaking internal arrays."""
    return {
        "agent1_integrity": dict(state.get("integrity", {})),
        "agent2_signal": _public_signal(dict(state.get("signal", {}))),
        "agent3_estimate": dict(state.get("estimate", {})),
        "agent4_decision": dict(state.get("decision", {})),
        "agent5_verifier": dict(state.get("quantum", {})),
        "agent6_hcqa": dict(state.get("circuit", {})),
        "agent7_output": dict(state.get("final_payload") or {}),
    }


def _make_analyze_response(payload: AnalyzeRequest, final_state: dict[str, Any], run_id: str) -> AnalyzeResponse:
    """Build the same response contract for the regular and streaming endpoints."""
    decision_data = final_state.get("decision", {})
    final_payload = final_state.get("final_payload") or {}
    dimensions = final_payload.get("dimensions")
    if not isinstance(dimensions, dict):
        dimensions = {}
    decision = decision_data.get("decision", "ABSTAIN")
    if decision not in {"CONTINUE", "ABSTAIN", "ESCALATE_NONLINEAR"}:
        decision = "ABSTAIN"
    capability_limits = list(final_state.get("capability_limits", []))
    if decision != "CONTINUE" and final_state.get("quantum", {}).get("status") == "skipped":
        capability_limits.append("Quantum simulation was not run because the orchestrator did not continue.")
    return AnalyzeResponse(
        run_id=run_id,
        shot_id=payload.shot_id,
        decision=decision,
        reason_codes=list(decision_data.get("reason_codes", ["DECISION_MISSING"])),
        stages=_public_stages(final_state),
        health_certificate=dimensions,
        llm_notes={str(key): str(value) for key, value in final_state.get("llm_notes", {}).items()},
        capability_limits=list(dict.fromkeys(capability_limits)),
        model_scope=str(final_payload.get("scope", "software diagnostics only; no hardware certification")),
    )


def _progress_message(agent: str, state: dict[str, Any]) -> dict[str, str]:
    """Translate actual completed agent outputs into concise user-facing progress."""
    if agent == "agent1_nerve":
        data = state.get("integrity", {})
        count = int(data.get("sample_count", 0) or 0)
        issues = data.get("issues", []) or []
        if data.get("valid"):
            message = f"Input integrity passed. The gate checked {count:,} samples and found no blocking metadata or waveform issues."
            status_text = "passed"
        else:
            readable = ", ".join(str(issue).replace("_", " ").lower() for issue in issues[:3]) or "an integrity check failed"
            message = f"Input integrity found a blocking issue: {readable}. The policy stage will decide whether to stop safely."
            status_text = "warning"
    elif agent == "agent2_kaiser":
        data = state.get("signal", {})
        metrics = data.get("metrics", {}) or {}
        if data.get("status") == "success":
            freq = metrics.get("dominant_frequency_hz")
            leak = metrics.get("spectral_leakage_ratio_outside_peak_bins")
            message = "Kaiser windowing and spectral diagnostics completed."
            if isinstance(freq, (int, float)):
                message += f" The strongest spectral bin is near {freq:.3g} Hz."
            if isinstance(leak, (int, float)):
                message += f" Leakage outside the peak band is {leak * 100:.2f}%."
            status_text = "passed"
        else:
            message, status_text = "Signal diagnostics did not complete successfully; the decision will remain fail-closed.", "warning"
    elif agent == "agent3_kalman":
        data = state.get("estimate", {})
        if data.get("status") == "success":
            err = data.get("phase_error_rad")
            unc = data.get("phase_uncertainty_rad")
            message = "The phase estimator completed its estimate."
            if isinstance(err, (int, float)):
                message += f" Absolute phase error is {abs(err):.4g} rad."
            if isinstance(unc, (int, float)):
                message += f" Estimated uncertainty is {unc:.4g} rad."
            status_text = "passed" if not data.get("failure_indicator", False) else "warning"
        else:
            message, status_text = "Phase estimation was skipped or failed; this is visible to the decision policy.", "warning"
    elif agent == "agent4_orchestrator":
        data = state.get("decision", {})
        decision = str(data.get("decision", "ABSTAIN"))
        messages = {
            "CONTINUE": "The policy found no blocking integrity or phase condition, so downstream verification may continue.",
            "ABSTAIN": "The policy chose to abstain because one or more required checks did not pass.",
            "ESCALATE_NONLINEAR": "The policy requested nonlinear escalation rather than treating this as a routine signal.",
        }
        message, status_text = messages.get(decision, "The policy returned an unknown result and will fail closed."), "passed" if decision == "CONTINUE" else "warning"
    elif agent == "agent6_hcqa":
        data = state.get("circuit", {})
        if data.get("status") == "success":
            depth = data.get("depth", "?")
            qfi = data.get("predicted_normalized_qfi")
            message = f"Circuit search completed at depth {depth}. This is a deterministic software search, not a hardware run."
            if isinstance(qfi, (int, float)):
                message += f" Predicted normalized QFI is {qfi:.3f}."
            status_text = "passed"
        else:
            message, status_text = "Circuit search was skipped because the policy stopped, or its fallback was used.", "muted" if data.get("status") == "skipped" else "warning"
    elif agent == "agent5_verifier":
        data = state.get("quantum", {})
        verification = data.get("verification_status", "not_evaluated")
        hw = data.get("hardware_status", "NOT_CONNECTED")
        if verification == "passed":
            message = f"The two-qubit simulator passed its deterministic invariants. Physical quantum hardware status is {hw.replace('_', ' ').lower()}."
            status_text = "passed"
        elif data.get("status") == "skipped":
            message, status_text = "Quantum verification was skipped because the policy did not continue.", "muted"
        else:
            message, status_text = "Quantum verification did not pass; the workflow must not claim hardware validation.", "warning"
    else:
        data = state.get("final_payload") or {}
        overall = str(data.get("overall_status", "ABSTAIN"))
        dims = data.get("dimensions", {}) or {}
        message = f"The final software diagnostic certificate is ready with {len(dims)} diagnostic dimensions. Overall decision: {overall}."
        message += " The certificate does not certify physical hardware health."
        status_text = "passed" if overall == "CONTINUE" else "warning"
    return {"agent": agent, "status": status_text, "message": message}


@router.get("/api/agents", response_model=list[AgentStatus], tags=["analysis"])
async def get_agent_status(_: bool = Depends(verify_token)) -> list[AgentStatus]:
    """Return clear roles and readiness; never return model IDs, keys or prompts."""
    statuses: list[AgentStatus] = []
    for definition in AGENT_REGISTRY:
        agent_id = definition["agent_id"]
        statuses.append(
            AgentStatus(
                agent_id=agent_id,
                name=definition["name"],
                role=definition["role"],
                status="ready",
                model_configured=(model_configured(agent_id) if definition["llm_optional"] else False),
                llm_optional=definition["llm_optional"],
                description=definition["description"],
            )
        )
    return statuses


@router.post("/api/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze_telemetry(
    payload: AnalyzeRequest,
    request: Request,
    _: bool = Depends(verify_token),
) -> AnalyzeResponse:
    """Run telemetry through the connected seven-agent pipeline and return the result."""
    run_id = str(uuid.uuid4())
    logger.info("Analysis started run_id=%s shot_id=%s", run_id, payload.shot_id)
    initial_state = initialize_state(payload.model_dump(mode="python"), run_id=run_id)
    engine = getattr(request.app.state, "qscope_engine", qscope_engine)
    try:
        final_state = await engine.ainvoke(initial_state)
    except Exception as exc:  # defensive boundary for unexpected orchestration defects
        logger.exception("Pipeline invocation failed run_id=%s (%s)", run_id, type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "PIPELINE_EXECUTION_FAILED", "run_id": run_id},
        ) from exc
    return _make_analyze_response(payload, final_state, run_id)


@router.post("/api/analyze/stream", tags=["analysis"])
async def analyze_telemetry_stream(
    payload: AnalyzeRequest,
    request: Request,
    _: bool = Depends(verify_token),
) -> StreamingResponse:
    """Run analysis and stream real completed-agent progress as Server-Sent Events."""
    run_id = str(uuid.uuid4())
    initial_state = initialize_state(payload.model_dump(mode="python"), run_id=run_id)
    engine = getattr(request.app.state, "qscope_engine", qscope_engine)

    async def event_stream():
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def progress_callback(name: str, state: dict[str, Any]) -> None:
            await queue.put(("progress", _progress_message(name, state)))

        async def run_pipeline() -> None:
            try:
                logger.info("Streaming analysis started run_id=%s shot_id=%s", run_id, payload.shot_id)
                state = await engine.ainvoke(initial_state, progress_callback=progress_callback)
                response = _make_analyze_response(payload, state, run_id)
                await queue.put(("complete", jsonable_encoder(response.model_dump(mode="json"))))
            except Exception as exc:
                logger.exception("Streaming pipeline failed run_id=%s (%s)", run_id, type(exc).__name__)
                await queue.put(("error", {"run_id": run_id, "message": "The analysis pipeline failed unexpectedly."}))
            finally:
                await queue.put(("end", {}))

        task = asyncio.create_task(run_pipeline())
        yield _sse("start", {"run_id": run_id, "message": "Q-SCOPE accepted the request. Live agent updates will appear as each stage finishes."})
        try:
            while True:
                event_name, data = await queue.get()
                if event_name == "end":
                    break
                yield _sse(event_name, data)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def _sse(event_name: str, data: Any) -> str:
    """Encode one standards-compliant Server-Sent Event."""
    safe_data = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return f"event: {event_name}\ndata: {safe_data}\n\n"


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz(request: Request) -> HealthResponse:
    """Liveness/readiness endpoint that does not disclose configuration secrets."""
    ready = getattr(request.app.state, "qscope_engine", None) is not None
    return HealthResponse(status="ok" if ready else "degraded", pipeline_ready=ready)


def _get_dataset_dir() -> Path:
    """Locate the dataset directory containing raw sensor telemetry CSVs."""
    candidates = [
        settings.project_root / "dataset",
        settings.project_root.parent / "dataset",
        Path("c:/Users/Slim5/Desktop/qscope/dataset"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return candidates[0]


def _load_dataset_samples(
    filename: str, channel: str = "x", offset: int = 0, count: int = 128
) -> tuple[list[float], float, int, list[str]]:
    """Read sensor samples from the specified CSV dataset file."""
    dataset_dir = _get_dataset_dir()
    safe_name = Path(filename).name
    target = dataset_dir / safe_name
    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sensor dataset file '{safe_name}' not found on server.",
        )

    samples: list[float] = []
    total_rows = 0
    sample_rate_hz = 200.0  # default 5ms cadence in ADXL345 logs
    channels: list[str] = ["x", "y", "z"]

    with open(target, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if fieldnames:
            channels = [col for col in fieldnames if col != "timestamp"]
        for idx, row in enumerate(reader):
            total_rows += 1
            if idx < offset:
                continue
            if len(samples) < count:
                try:
                    if channel == "magnitude":
                        vx = float(row.get("x", 0.0) or 0.0)
                        vy = float(row.get("y", 0.0) or 0.0)
                        vz = float(row.get("z", 0.0) or 0.0)
                        val = math.sqrt(vx * vx + vy * vy + vz * vz)
                    else:
                        val = float(row.get(channel, row.get("x", 0.0)) or 0.0)
                    samples.append(val)
                except (ValueError, TypeError):
                    samples.append(0.0)

    return samples, sample_rate_hz, total_rows, channels


def _build_analyze_request_from_dataset(payload: DatasetAnalyzeRequest) -> AnalyzeRequest:
    """Convert dataset slicing parameters into a validated AnalyzeRequest."""
    samples, sample_rate_hz, total_rows, _ = _load_dataset_samples(
        filename=payload.dataset_file,
        channel=payload.channel,
        offset=payload.offset,
        count=payload.sample_count,
    )
    if len(samples) < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Offset {payload.offset} in '{payload.dataset_file}' only yielded {len(samples)} samples (at least 16 required).",
        )

    dt = len(samples) / sample_rate_hz
    return AnalyzeRequest(
        shot_id=payload.shot_id,
        sensor_trace_id=f"{payload.sensor_trace_id}-{payload.channel.upper()}",
        calibration_version="v2.1.0",
        calibration_valid=True,
        population_ratio_interval_s=(0.0, dt),
        laser_pulse_times_s=(0.0, dt / 2.0, dt),
        samples=samples,
        sampling_rate_hz=sample_rate_hz,
        carrier_frequency_hz=payload.carrier_frequency_hz,
        phase_reference_rad=0.0,
        phase_tolerance_rad=payload.phase_tolerance_rad,
        max_phase_uncertainty_rad=payload.max_phase_uncertainty_rad,
        kaiser_beta=payload.kaiser_beta,
        signal_coefficients=(1.0, 1.0),
        pulse_graph_nodes=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ],
        pulse_graph_edges=[[0, 1], [1, 2], [2, 3]],
        sensitivity_function=[0.0, 0.1, 0.2, 0.3, 0.4],
        max_gate_depth=payload.max_gate_depth,
        target_normalized_qfi=payload.target_normalized_qfi,
    )


@router.get("/api/datasets", response_model=list[DatasetSummary], tags=["datasets"])
async def list_datasets(_: bool = Depends(verify_token)) -> list[DatasetSummary]:
    """List available raw sensor telemetry CSV datasets."""
    dataset_dir = _get_dataset_dir()
    summaries: list[DatasetSummary] = []
    if not dataset_dir.exists():
        return summaries

    for item in sorted(dataset_dir.glob("*.csv")):
        safe_name = item.name
        try:
            samples_x, rate, total_rows, channels = _load_dataset_samples(safe_name, "x", 0, 64)
            samples_y, _, _, _ = _load_dataset_samples(safe_name, "y", 0, 64)
            samples_z, _, _, _ = _load_dataset_samples(safe_name, "z", 0, 64)
            name_label = safe_name.replace(".csv", "").replace("_", " ").title()
            summaries.append(
                DatasetSummary(
                    filename=safe_name,
                    name=name_label,
                    row_count=total_rows,
                    channels=channels,
                    sample_rate_hz=rate,
                    preview_samples={
                        "x": samples_x[:32],
                        "y": samples_y[:32],
                        "z": samples_z[:32],
                    },
                )
            )
        except Exception as exc:
            logger.warning("Could not summarize dataset %s: %s", safe_name, exc)
    return summaries


@router.get("/api/datasets/{filename}", tags=["datasets"])
async def get_dataset_data(
    filename: str,
    channel: str = "x",
    offset: int = 0,
    count: int = 256,
    _: bool = Depends(verify_token),
) -> dict[str, Any]:
    """Retrieve raw telemetry samples for visualization."""
    samples, sample_rate_hz, total_rows, channels = _load_dataset_samples(
        filename=filename, channel=channel, offset=offset, count=min(count, 4096)
    )
    return {
        "filename": filename,
        "channel": channel,
        "offset": offset,
        "count": len(samples),
        "total_rows": total_rows,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "samples": samples,
    }


@router.post("/api/analyze/from-dataset", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze_from_dataset(
    payload: DatasetAnalyzeRequest,
    request: Request,
    _: bool = Depends(verify_token),
) -> AnalyzeResponse:
    """Slice real sensor CSV telemetry and execute the seven-agent pipeline."""
    analyze_req = _build_analyze_request_from_dataset(payload)
    run_id = str(uuid.uuid4())
    logger.info(
        "Dataset analysis started run_id=%s file=%s channel=%s",
        run_id,
        payload.dataset_file,
        payload.channel,
    )
    initial_state = initialize_state(analyze_req.model_dump(mode="python"), run_id=run_id)
    engine = getattr(request.app.state, "qscope_engine", qscope_engine)
    try:
        final_state = await engine.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Dataset pipeline invocation failed run_id=%s (%s)", run_id, type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "DATASET_PIPELINE_EXECUTION_FAILED", "run_id": run_id},
        ) from exc
    return _make_analyze_response(analyze_req, final_state, run_id)


@router.post("/api/analyze/from-dataset/stream", tags=["analysis"])
async def analyze_from_dataset_stream(
    payload: DatasetAnalyzeRequest,
    request: Request,
    _: bool = Depends(verify_token),
) -> StreamingResponse:
    """Stream 7-agent execution progress for real dataset telemetry via SSE."""
    analyze_req = _build_analyze_request_from_dataset(payload)
    run_id = str(uuid.uuid4())
    initial_state = initialize_state(analyze_req.model_dump(mode="python"), run_id=run_id)
    engine = getattr(request.app.state, "qscope_engine", qscope_engine)

    async def event_stream():
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def progress_callback(name: str, state: dict[str, Any]) -> None:
            await queue.put(("progress", _progress_message(name, state)))

        async def run_pipeline() -> None:
            try:
                logger.info(
                    "Streaming dataset analysis started run_id=%s file=%s channel=%s",
                    run_id,
                    payload.dataset_file,
                    payload.channel,
                )
                state = await engine.ainvoke(initial_state, progress_callback=progress_callback)
                response = _make_analyze_response(analyze_req, state, run_id)
                await queue.put(("complete", jsonable_encoder(response.model_dump(mode="json"))))
            except Exception as exc:
                logger.exception("Streaming dataset pipeline failed run_id=%s (%s)", run_id, type(exc).__name__)
                await queue.put(("error", {"run_id": run_id, "message": "The analysis pipeline failed unexpectedly."}))
            finally:
                await queue.put(("end", {}))

        task = asyncio.create_task(run_pipeline())
        yield _sse(
            "start",
            {
                "run_id": run_id,
                "message": f"Telemetry loaded from '{payload.dataset_file}' ({payload.channel}-axis). Processing across 7 agents...",
            },
        )
        try:
            while True:
                event_name, data = await queue.get()
                if event_name == "end":
                    break
                yield _sse(event_name, data)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


# =====================================================================
# AZHVA AI DIAGNOSTIC ENDPOINTS
# =====================================================================

@router.post("/api/azhva/translate", response_model=AzhvaTranslateResponse, tags=["azhva"])
async def azhva_translate(payload: AzhvaTranslateRequest) -> AzhvaTranslateResponse:
    """Azhva Phase 1: Machine-to-human analog telemetry spectrogram translation."""
    result = await run_phase_1_translation(payload.samples, payload.sampling_rate_hz)
    return AzhvaTranslateResponse(
        machine_voice=result["machine_voice"],
        perception_notes=result["perception_notes"],
        spectrogram_base64=result["spectrogram_base64"],
    )


@router.post("/api/azhva/chat", response_model=AzhvaChatResponse, tags=["azhva"])
async def azhva_chat(payload: AzhvaChatRequest) -> AzhvaChatResponse:
    """Azhva Phase 2: Autonomous flight operations copilot conversational agent."""
    reply = await run_phase_2_chat(payload.message, payload.session_id)
    return AzhvaChatResponse(reply=reply)




# =====================================================================
# PERSISTENT DATABASE ROUTES (PostgreSQL / SQLite)
# =====================================================================

@router.get("/api/projects", response_model=list[dict[str, Any]], tags=["projects"])
def list_projects(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """List all user projects persisted in the database."""
    try:
        items = db.query(ProjectModel).order_by(ProjectModel.created_at.desc()).all()
        return [p.to_dict() for p in items]
    except Exception as exc:
        logger.error("Failed to read projects from database: %s", exc)
        return []


@router.post("/api/projects", response_model=dict[str, Any], tags=["projects"])
def save_or_update_project(
    payload: ProjectCreateRequest,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Persist or update a flight project with instrument configs and runtimes."""
    from datetime import datetime
    try:
        existing = db.query(ProjectModel).filter(ProjectModel.id == payload.id).first()
        runtimes_str = json.dumps(payload.runtimes or [])
        if existing:
            existing.name = payload.name
            if payload.sensor: existing.sensor = payload.sensor
            if payload.device: existing.device = payload.device
            if payload.sensorConnected is not None: existing.sensor_connected = payload.sensorConnected
            if payload.gps: existing.gps = payload.gps
            if payload.gpsConnected is not None: existing.gps_connected = payload.gpsConnected
            if payload.backend: existing.backend = payload.backend
            if payload.dataset: existing.dataset = payload.dataset
            existing.runtimes_json = runtimes_str
            existing.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            logger.info("Updated project id=%s name='%s'", existing.id, existing.name)
            return existing.to_dict()
        else:
            new_p = ProjectModel(
                id=payload.id,
                name=payload.name,
                sensor=payload.sensor or "ADXL345 Tri-Axial Accelerometer (Real Telemetry)",
                device=payload.device or "Sensor Gateway / Serial Bridge",
                sensor_connected=payload.sensorConnected if payload.sensorConnected is not None else True,
                gps=payload.gps or "USB / Serial GPS receiver",
                gps_connected=payload.gpsConnected if payload.gpsConnected is not None else True,
                backend=payload.backend or "Aer Simulator",
                dataset=payload.dataset or "qscope_clean_synthetic_adxl345.csv",
                created=payload.created or datetime.utcnow().strftime("%d %b %Y"),
                runtimes_json=runtimes_str,
            )
            db.add(new_p)
            db.commit()
            db.refresh(new_p)
            logger.info("Created new project id=%s name='%s'", new_p.id, new_p.name)
            return new_p.to_dict()
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to persist project id=%s: %s", payload.id, exc)
        raise HTTPException(status_code=500, detail="Database persistence failed") from exc


@router.delete("/api/projects/{project_id}", tags=["projects"])
def delete_project_by_id(project_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Delete a project and its runtimes from the database."""
    try:
        item = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if item:
            db.delete(item)
            db.commit()
            logger.info("Deleted project id=%s", project_id)
        return {"status": "success", "deleted_id": project_id}
    except Exception as exc:
        db.rollback()
        logger.error("Failed to delete project id=%s: %s", project_id, exc)
        raise HTTPException(status_code=500, detail="Database delete failed") from exc


@router.get("/api/history", response_model=list[dict[str, Any]], tags=["history"])
def get_analysis_history(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """Retrieve recent telemetry analysis runs from database."""
    try:
        items = db.query(AnalysisRunModel).order_by(AnalysisRunModel.created_at.desc()).limit(50).all()
        return [r.to_dict() for r in items]
    except Exception as exc:
        logger.error("Failed to fetch history: %s", exc)
        return []
