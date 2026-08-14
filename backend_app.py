"""
OKUMA CNC Predictive Maintenance — FastAPI Backend (spec section 28 + energy update)
========================================================================================
STATUS: the original routes below (/health, /live/*, /history*, /compare, /explain,
/maintenance) have since been run for real by the user and confirmed working,
including after the score_batch() perf fix — see CHANGELOG.md.

The /energy/* routes follow the exact same pattern as the already-working
/history/trend (vectorized batch scoring, not a per-row loop — see
score_energy_batch() in prediction_engine.py).

The /live/demo/* routes are DEMO REPLAY MODE — see the docstring on
/live/predict below and README_LIVE_DEMO.md.

The /maintenance/events routes are new: a real, persisted maintenance-event
log. This is not an RUL model — it is the prerequisite for one. See its
docstrings and README_LIVE_DEMO.md ("gap 3: RUL") for why.

Security: CORS origins and an optional API key are both read from environment
variables (see below) rather than hardcoded, so this can be tightened for a
real deployment without editing code. Defaults stay permissive for local dev.
"""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, List
from datetime import datetime, timezone

from prediction_engine import PredictionEngine

app = FastAPI(title="OKUMA CNC Predictive Maintenance + Energy API", version="0.3.0")

# ---------------------------------------------------------------------------
# Security config — read from environment, not hardcoded.
#
# ALLOWED_ORIGINS: comma-separated list of allowed frontend origins, e.g.
#   ALLOWED_ORIGINS="https://your-dashboard.vercel.app,http://localhost:3000"
# Defaults to "*" (any origin) ONLY because that's the right default for a
# laptop demo against localhost -- set this explicitly before deploying
# anywhere reachable from outside your own machine.
#
# API_KEY: if set, every route below except /health requires a matching
# `X-API-Key` header. Unset (default) = no auth, matching the "no network
# access to test auth flows" constraint this project was built under -- set
# this before deploying anywhere reachable from outside your own machine.
# ---------------------------------------------------------------------------
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = ["*"] if _allowed_origins_env.strip() == "*" else [
    o.strip() for o in _allowed_origins_env.split(",") if o.strip()
]
API_KEY = os.environ.get("API_KEY")  # None = auth disabled

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """FastAPI dependency: no-op if API_KEY isn't set (local dev default).
    Once API_KEY is set in the environment, every protected route 401s
    without a matching X-API-Key header."""
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


engine = PredictionEngine()


class HistoryPredictRequest(BaseModel):
    timestamp: str  # ISO format, e.g. "2026-04-23T00:05:00"


class CompareRequest(BaseModel):
    timestamps: List[str]  # 2-5 timestamps


class EnergyAnalyzeRequest(BaseModel):
    timestamp: str


@app.get("/health")
def health():
    """Liveness check — does the API process itself respond."""
    return {"status": "ok", "engine_loaded": engine is not None}


@app.get("/live/status", dependencies=[Depends(require_api_key)])
def live_status():
    """
    Per project rule: never fake live sensor values. No PLC/live connection
    exists in this deployment, so this is always reported honestly.
    """
    return {"live_sensor": "NOT CONNECTED", "message": "No live PLC/sensor feed is configured. "
             "Use /history/predict for historical data, or /live/predict in DEMO REPLAY MODE only."}


@app.post("/live/predict", dependencies=[Depends(require_api_key)])
def live_predict():
    """
    No live connection exists. This endpoint intentionally returns an honest
    'not connected' response rather than a DEMO REPLAY — wiring in a replay
    mode is a deliberate choice for whoever deploys this, not a default,
    since a replay could be mistaken for real live data if mislabeled.
    """
    raise HTTPException(status_code=503, detail="LIVE SENSOR: NOT CONNECTED. "
                         "No live PLC/sensor feed is configured for this deployment.")


# ---------------------------------------------------------------------------
# DEMO REPLAY MODE — an explicit, separately namespaced feature for showing
# the dashboard to a client before a real PLC/sensor feed exists. This is the
# "wiring in a replay mode is a deliberate choice" case /live/predict's
# docstring above refers to: nothing here is fabricated, it's the same
# tested prediction_engine.predict_historical() replaying real historical
# readings in order. Every response carries an explicit DEMO_REPLAY label so
# it can never be confused with /live/predict's real-time (currently 503)
# contract. See LIMITATIONS_AND_STATUS.md, section 27.
# ---------------------------------------------------------------------------
DEMO_LABEL = ("SIMULATED LIVE FEED \u2014 replaying real historical sensor data in chronological "
              "order. No live PLC/sensor connection exists for this deployment.")

# Sampled every 4 hours from the 5-minute source data so each "tick" looks
# meaningfully different (matches the cadence used in dashboard_demo/) rather
# than replaying near-identical consecutive 5-minute readings.
_demo_ts_mask = (engine.features["timestamp"].dt.minute == 0) & (engine.features["timestamp"].dt.hour % 4 == 0)
DEMO_TIMESTAMPS = sorted(engine.features.loc[_demo_ts_mask, "timestamp"].tolist())


@app.get("/live/demo/info", dependencies=[Depends(require_api_key)])
def live_demo_info():
    """Metadata for the demo replay stream: how many points, what date range."""
    return {
        "mode": "DEMO_REPLAY",
        "label": DEMO_LABEL,
        "total_points": len(DEMO_TIMESTAMPS),
        "date_range": {"from": str(DEMO_TIMESTAMPS[0]), "to": str(DEMO_TIMESTAMPS[-1])},
        "sample_interval": "4 hours (sampled down from the 5-minute source data for demo pacing)",
    }


@app.get("/live/demo/next", dependencies=[Depends(require_api_key)])
def live_demo_next(cursor: int = -1):
    """
    Stateless "next demo reading" step. The caller (frontend) owns the cursor
    and passes it back each call so multiple demo viewers never collide on
    shared server-side state; this returns the following point plus the new
    cursor, looping back to the start at the end of the window. Used to drive
    both the fast-playback and real-time-paced demo modes -- the frontend
    just controls how often it calls this.
    """
    total = len(DEMO_TIMESTAMPS)
    next_idx = (cursor + 1) % total
    ts = DEMO_TIMESTAMPS[next_idx]
    reading = engine.predict_historical(ts)
    return {
        "mode": "DEMO_REPLAY",
        "label": DEMO_LABEL,
        "cursor": next_idx,
        "total_points": total,
        "looped": next_idx == 0 and cursor != -1,
        "reading": reading,
    }


@app.get("/live/demo/at", dependencies=[Depends(require_api_key)])
def live_demo_at(cursor: int):
    """Fetch a specific demo point by cursor index without advancing -- used
    by manual-step mode's back/forward controls."""
    total = len(DEMO_TIMESTAMPS)
    idx = cursor % total
    ts = DEMO_TIMESTAMPS[idx]
    reading = engine.predict_historical(ts)
    return {
        "mode": "DEMO_REPLAY",
        "label": DEMO_LABEL,
        "cursor": idx,
        "total_points": total,
        "reading": reading,
    }


@app.get("/history", dependencies=[Depends(require_api_key)])
def history_range():
    """Returns the available historical date range the client can query."""
    t_min = engine.features["timestamp"].min()
    t_max = engine.features["timestamp"].max()
    return {"available_from": str(t_min), "available_to": str(t_max),
            "sampling_interval": "5 minutes (nominal)"}


@app.post("/history/predict", dependencies=[Depends(require_api_key)])
def history_predict(req: HistoryPredictRequest):
    try:
        ts = datetime.fromisoformat(req.timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be ISO format, e.g. 2026-04-23T00:05:00")
    return engine.predict_historical(ts)


@app.get("/history/trend", dependencies=[Depends(require_api_key)])
def history_trend(start: str, end: str):
    """Health Index trend between two timestamps, for the historical dashboard's trend chart."""
    try:
        start_ts, end_ts = datetime.fromisoformat(start), datetime.fromisoformat(end)
    except ValueError:
        raise HTTPException(status_code=400, detail="start/end must be ISO format")
    active = engine.features[(engine.features["timestamp"] >= start_ts) &
                              (engine.features["timestamp"] <= end_ts) &
                              (engine.features["operating_state"] == "ACTIVE")]

    scored = engine.score_batch(active)  # vectorized -- see prediction_engine.py score_batch()
    merged = active.loc[scored.index, ["timestamp"]].copy()
    merged["health_index"] = scored["health_index"].round(1)

    # Downsample for chart legibility/payload size -- a 2-month range can have
    # 10k+ points, which is more than a line chart needs and slows the browser
    # for no visual benefit. Caps at ~500 points, evenly spaced.
    max_points = 500
    if len(merged) > max_points:
        step = len(merged) // max_points
        merged = merged.iloc[::step]

    points = [{"timestamp": str(r.timestamp), "health_index": float(r.health_index)} for r in merged.itertuples()]
    return {"label": "OBSERVED HISTORICAL DATA", "points": points}


@app.post("/compare", dependencies=[Depends(require_api_key)])
def compare(req: CompareRequest):
    if not (2 <= len(req.timestamps) <= 5):
        raise HTTPException(status_code=400, detail="Provide between 2 and 5 timestamps")
    results = []
    for ts_str in req.timestamps:
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid timestamp: {ts_str}")
        results.append(engine.predict_historical(ts))
    return {"comparisons": results}


@app.get("/explain", dependencies=[Depends(require_api_key)])
def explain(timestamp: str):
    try:
        ts = datetime.fromisoformat(timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be ISO format")
    result = engine.predict_historical(ts)
    return {
        "timestamp": result["timestamp_matched"],
        "health_index": result["health_index"],
        "top_parameters": result["top_parameters"],
        "interpretation_note": "Feature importance reflects statistical association with the "
                                "anomaly score, not confirmed physical causality.",
    }


@app.get("/maintenance", dependencies=[Depends(require_api_key)])
def maintenance(timestamp: str):
    try:
        ts = datetime.fromisoformat(timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be ISO format")
    result = engine.predict_historical(ts)
    return {
        "timestamp": result["timestamp_matched"],
        "priority": result["priority"],
        "maintenance_category": result["maintenance_category"],
        "recommendation": result["recommendation"],
        "maintenance_window": result.get("maintenance_window"),
        "predicted_maintenance_datetime": result["predicted_maintenance_datetime"],
        "confidence": result["confidence"],
    }


# ---------------------------------------------------------------------------
# Energy endpoints (spec Part 26). All power values are apparent power (kVA)
# -- no power factor is available in this dataset, so this is explicitly NOT
# true active power (kW). See energy_baseline.json for the full derivation
# and its documented limitations (no RPM/spindle-load signal available).
# ---------------------------------------------------------------------------

@app.get("/energy/current", dependencies=[Depends(require_api_key)])
def energy_current():
    """Most recent available reading — there's no live feed, so this is the
    latest point in the historical dataset, clearly labeled as such."""
    latest_ts = engine.features["timestamp"].max()
    result = engine.predict_historical(latest_ts)
    return {
        "as_of": result["timestamp_matched"],
        "note": "No live sensor feed is connected — this is the most recent HISTORICAL reading, not real-time.",
        "actual_power_kva": result["actual_power_kva"],
        "expected_power_kva": result["expected_power_kva"],
        "power_deviation_kva": result["power_deviation_kva"],
        "energy_excess_percent": result["energy_excess_percent"],
        "energy_status": result["energy_status"],
        "machine_state": result["machine_state"],
    }


@app.get("/energy/trend", dependencies=[Depends(require_api_key)])
def energy_trend(start: str, end: str):
    """Actual vs expected power over a date range, for the energy trend chart."""
    try:
        start_ts, end_ts = datetime.fromisoformat(start), datetime.fromisoformat(end)
    except ValueError:
        raise HTTPException(status_code=400, detail="start/end must be ISO format")

    window = engine.features[(engine.features["timestamp"] >= start_ts) & (engine.features["timestamp"] <= end_ts)]
    scored = engine.score_energy_batch(window)  # vectorized -- see CHANGELOG on why this matters
    merged = window.loc[scored.index, ["timestamp", "operating_state"]].join(scored)

    max_points = 500  # same rationale as /history/trend -- chart legibility, not raw dump
    if len(merged) > max_points:
        step = len(merged) // max_points
        merged = merged.iloc[::step]

    points = [
        {"timestamp": str(r.timestamp), "actual_power_kva": float(r.actual_power_kva),
         "expected_power_kva": float(r.expected_power_kva), "energy_status": r.energy_status}
        for r in merged.itertuples()
    ]
    return {"label": "OBSERVED HISTORICAL DATA", "power_unit": "apparent power (kVA), not true kW", "points": points}


@app.post("/energy/analyze", dependencies=[Depends(require_api_key)])
def energy_analyze(req: EnergyAnalyzeRequest):
    """Energy-focused view of a single timestamp — same underlying engine call
    as /history/predict, just returns only the energy-relevant fields."""
    try:
        ts = datetime.fromisoformat(req.timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be ISO format")
    result = engine.predict_historical(ts)
    return {
        "timestamp_requested": result["timestamp_requested"],
        "timestamp_matched": result["timestamp_matched"],
        "machine_state": result["machine_state"],
        "actual_power_kva": result["actual_power_kva"],
        "expected_power_kva": result["expected_power_kva"],
        "power_deviation_kva": result["power_deviation_kva"],
        "energy_excess_percent": result["energy_excess_percent"],
        "energy_status": result["energy_status"],
        "top_energy_drivers": result["top_energy_drivers"],
        "energy_recommendation": result["energy_recommendation"],
        "joint_flag": result["joint_flag"],
        "joint_explanation": result["joint_explanation"],
    }


@app.get("/energy/opportunities", dependencies=[Depends(require_api_key)])
def energy_opportunities(start: str = None, end: str = None, limit: int = 50):
    """Lists ACTIVE-state readings flagged ENERGY_SAVING_OPPORTUNITY within a
    range (defaults to the full dataset), most recent first."""
    window = engine.features
    if start:
        try:
            window = window[window["timestamp"] >= datetime.fromisoformat(start)]
        except ValueError:
            raise HTTPException(status_code=400, detail="start must be ISO format")
    if end:
        try:
            window = window[window["timestamp"] <= datetime.fromisoformat(end)]
        except ValueError:
            raise HTTPException(status_code=400, detail="end must be ISO format")

    scored = engine.score_energy_batch(window)
    merged = window.loc[scored.index, ["timestamp"]].join(scored)
    opportunities = merged[merged["energy_status"] == "ENERGY_SAVING_OPPORTUNITY"].sort_values("timestamp", ascending=False)

    results = []
    for r in opportunities.head(limit).itertuples():
        detail = engine.predict_historical(r.timestamp)
        results.append({
            "timestamp": str(r.timestamp),
            "actual_power_kva": float(r.actual_power_kva),
            "expected_power_kva": float(r.expected_power_kva),
            "energy_excess_percent": float(r.energy_excess_percent),
            "top_energy_drivers": detail["top_energy_drivers"],
            "energy_recommendation": detail["energy_recommendation"],
        })
    return {
        "count": len(results),
        "total_matching_in_range": len(opportunities),
        "note": "estimated_saving is intentionally not included -- see spec Part 30: actual savings can "
                "only be confirmed with post-action measured energy data, which doesn't exist yet.",
        "opportunities": results,
    }


# ---------------------------------------------------------------------------
# Maintenance event log — the actual prerequisite for a real RUL model.
#
# This is NOT an RUL model and does not compute one. predicted_maintenance_
# datetime and rul stay null everywhere else in this API, honestly, because
# no failure/maintenance history exists in the training data (see
# LIMITATIONS_AND_STATUS.md). This is the mechanism for fixing that: every
# real maintenance/failure/inspection event logged here becomes a labeled
# example. Once enough real events exist, THAT is the point where training
# an actual time-to-failure model against the existing Health Index history
# becomes possible -- and honest. Nothing here fabricates readiness before
# that; /maintenance/events/summary reports a real count against a stated
# rule-of-thumb threshold, not a promise.
#
# Persisted to a local SQLite file so entries survive a server restart.
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "data" / "maintenance_events.db"
EVENT_TYPES = ["FAILURE", "CORRECTIVE_MAINTENANCE", "PREVENTIVE_MAINTENANCE", "INSPECTION", "SENSOR_SERVICE", "OTHER"]
KNOWN_SUBSYSTEMS = ["Axis servo drives", "Central lubrication", "Control panel / ambient", "Hydraulic system",
                    "Mist lubrication", "Power / stabilizer", "Spindle", "Spindle cooling", "Tool magazine"]
MIN_EVENTS_RULE_OF_THUMB = 15  # a rule of thumb, not a guarantee of model quality -- see summary note below


@contextmanager
def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                subsystem TEXT,
                description TEXT,
                logged_by TEXT,
                logged_at TEXT NOT NULL
            )
        """)
        yield conn
        conn.commit()
    finally:
        conn.close()


class MaintenanceEventCreate(BaseModel):
    event_timestamp: str  # ISO timestamp of when the event actually happened on the machine
    event_type: str
    subsystem: Optional[str] = None
    description: Optional[str] = None
    logged_by: Optional[str] = None

    @field_validator("event_type")
    @classmethod
    def _valid_type(cls, v):
        if v not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {EVENT_TYPES}")
        return v

    @field_validator("event_timestamp")
    @classmethod
    def _valid_ts(cls, v):
        datetime.fromisoformat(v)  # raises ValueError -> 422 if malformed
        return v


@app.post("/maintenance/events", dependencies=[Depends(require_api_key)])
def log_maintenance_event(req: MaintenanceEventCreate):
    """Log a real maintenance/failure/inspection event. This is additive --
    nothing else in the API reads or reacts to these yet. See the module
    docstring above for why this exists and what it doesn't do."""
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO maintenance_events (event_timestamp, event_type, subsystem, description, logged_by, logged_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (req.event_timestamp, req.event_type, req.subsystem, req.description, req.logged_by,
             datetime.now(timezone.utc).isoformat()),
        )
        new_id = cur.lastrowid
    return {"id": new_id, "status": "logged"}


@app.get("/maintenance/events", dependencies=[Depends(require_api_key)])
def list_maintenance_events(start: str = None, end: str = None, limit: int = 200):
    if start:
        try:
            datetime.fromisoformat(start)
        except ValueError:
            raise HTTPException(status_code=400, detail="start must be ISO format")
    if end:
        try:
            datetime.fromisoformat(end)
        except ValueError:
            raise HTTPException(status_code=400, detail="end must be ISO format")

    query = "SELECT * FROM maintenance_events WHERE 1=1"
    params = []
    if start:
        query += " AND event_timestamp >= ?"
        params.append(start)
    if end:
        query += " AND event_timestamp <= ?"
        params.append(end)
    query += " ORDER BY event_timestamp DESC LIMIT ?"
    params.append(limit)

    with _db() as conn:
        rows = conn.execute(query, params).fetchall()
    return {"count": len(rows), "events": [dict(r) for r in rows]}


@app.delete("/maintenance/events/{event_id}", dependencies=[Depends(require_api_key)])
def delete_maintenance_event(event_id: int):
    """For fixing mis-logged entries -- e.g. a wrong timestamp typed during a live demo."""
    with _db() as conn:
        cur = conn.execute("DELETE FROM maintenance_events WHERE id = ?", (event_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"No event with id {event_id}")
    return {"status": "deleted", "id": event_id}


@app.get("/maintenance/events/summary", dependencies=[Depends(require_api_key)])
def maintenance_events_summary():
    """
    Honest status report on RUL-model readiness -- NOT a model, NOT a
    prediction. Reports how many real events are logged so far against a
    stated rule-of-thumb threshold. Crossing the threshold doesn't trigger
    anything automatically; it means training a first real time-to-failure
    model against the existing Health Index history becomes a reasonable
    next engineering step, to be done deliberately, not a promise about
    what accuracy it would achieve.
    """
    with _db() as conn:
        rows = conn.execute("SELECT event_type, event_timestamp FROM maintenance_events").fetchall()

    total = len(rows)
    by_type = {}
    for r in rows:
        by_type[r["event_type"]] = by_type.get(r["event_type"], 0) + 1
    timestamps = sorted(r["event_timestamp"] for r in rows)

    return {
        "total_events": total,
        "by_type": by_type,
        "date_range": {"from": timestamps[0], "to": timestamps[-1]} if timestamps else None,
        "min_events_rule_of_thumb": MIN_EVENTS_RULE_OF_THUMB,
        "rul_model_readiness": "READY_FOR_FIRST_PASS" if total >= MIN_EVENTS_RULE_OF_THUMB else "NOT_READY",
        "note": (
            f"{total}/{MIN_EVENTS_RULE_OF_THUMB} events logged toward a rule-of-thumb threshold for "
            "attempting a first real RUL model. This is a rule of thumb, not a guarantee -- model "
            "quality will also depend on how varied the logged failure modes/subsystems are, not just "
            "the count. predicted_maintenance_datetime and rul stay null everywhere in this API until "
            "that model actually exists and is validated."
        ),
    }

