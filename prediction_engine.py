"""
OKUMA CNC Predictive Maintenance — Production Prediction Engine (spec section 22-23)
========================================================================================
Single reusable module. Runs outside Jupyter. Two entry points:

  - predict_historical(requested_timestamp): the client-facing mode — look up
    the nearest available historical record and run the full pipeline on it.
    This is what the dashboard's "pick a date/time" feature calls.
  - predict_from_raw(sensor_row, recent_history_df): for a genuinely live
    reading, given a buffer of recent same-session history to compute rolling
    features from. Architecturally supported; not exercised with real live
    data since no live sensor/PLC connection exists (see LIVE SENSOR status).

Every output follows the STANDARD MODEL OUTPUT schema (section 23) — unsupported
fields are null, nothing is invented.
"""
import pandas as pd
import numpy as np
import json, pickle
from pathlib import Path

CFG_DIR = "configs"
MODEL_DIR = "models"
FEAT_DIR = "data/features"
OUT_DIR = "data/processed"


class PredictionEngine:
    def __init__(self):
        with open(f"{CFG_DIR}/feature_columns.json") as fh:
            self.feature_meta = json.load(fh)
        with open(f"{CFG_DIR}/selected_features.json") as fh:
            self.selected_features = json.load(fh)["selected_features"]
        with open(f"{CFG_DIR}/thresholds.json") as fh:
            self.thresholds = json.load(fh)
        with open(f"{CFG_DIR}/parameter_to_subsystem.json") as fh:
            self.subsystem_map = json.load(fh)
        with open(f"{MODEL_DIR}/top_parameters_report.json") as fh:
            self.top_params_global = json.load(fh)["top_parameters"]
        with open(f"{CFG_DIR}/energy_baseline.json") as fh:
            self.energy_baseline = json.load(fh)["baseline_by_state"]
        with open(f"{CFG_DIR}/energy_thresholds.json") as fh:
            self.energy_thresholds = json.load(fh)
        with open(f"{MODEL_DIR}/energy_drivers_report.json") as fh:
            self.energy_drivers_global = json.load(fh)["top_candidates"]

        with open(f"{MODEL_DIR}/isolation_forest_v3.pkl", "rb") as fh:
            self.iso = pickle.load(fh)
        with open(f"{MODEL_DIR}/scaler_v3.pkl", "rb") as fh:
            self.scaler = pickle.load(fh)
        with open(f"{MODEL_DIR}/ambient_adjustment_models_v2.pkl", "rb") as fh:
            self.ambient = pickle.load(fh)

        self.ambient_cols = [c for c in self.selected_features
                              if c.startswith("Panel Temp") or c.startswith("Panel Humidity")]
        self.machine_features = [c for c in self.selected_features if c not in self.ambient_cols]

        # Health Index calibration (train raw-score range) — must match training exactly
        train_hi = pd.read_csv(f"{MODEL_DIR}/health_index_v3_train.csv")
        # recover calibration bounds by inverting: stored health_index and raw_anomaly_score both present
        self._hi_score_min = self._infer_calibration(train_hi)

        self.features = pd.read_csv(f"{FEAT_DIR}/features_full.csv", parse_dates=["timestamp"])
        ambient_24h = pd.read_csv(f"{OUT_DIR}/ambient_24h_calendar.csv", parse_dates=["timestamp"])
        self.features = self.features.merge(ambient_24h, on="timestamp", how="left")
        self.features = self.features.sort_values("timestamp").reset_index(drop=True)

        active = self.features[self.features["operating_state"] == "ACTIVE"]
        self.active_baseline = active  # used for baseline means in per-instance deviation reporting

    @staticmethod
    def _infer_calibration(train_hi_df):
        # health_index = clip((raw - smin)/(smax-smin)*100, 0, 100) -> recover smin/smax from two points
        row_lo = train_hi_df.loc[train_hi_df["health_index"].idxmin()]
        row_hi = train_hi_df.loc[train_hi_df["health_index"].idxmax()]
        return {"example_lo": (row_lo["raw_anomaly_score"], row_lo["health_index"]),
                "example_hi": (row_hi["raw_anomaly_score"], row_hi["health_index"])}

    def _build_ambient_design(self, row):
        t, h = row["Panel Temp"], row["Panel Humidity"]
        t4h, h4h = row["Panel Temp_mean_4h"], row["Panel Humidity_mean_4h"]
        t24h, h24h = row["Panel_Temp_mean_24h_calendar"], row["Panel_Humidity_mean_24h_calendar"]
        return np.array([[t, h, t**2, h**2, t*h, t4h, h4h, t24h, h24h]])

    def _build_ambient_design_batch(self, df):
        t, h = df["Panel Temp"].values, df["Panel Humidity"].values
        t4h, h4h = df["Panel Temp_mean_4h"].values, df["Panel Humidity_mean_4h"].values
        t24h, h24h = df["Panel_Temp_mean_24h_calendar"].values, df["Panel_Humidity_mean_24h_calendar"].values
        return np.column_stack([t, h, t**2, h**2, t * h, t4h, h4h, t24h, h24h])

    def score_batch(self, df):
        """
        Vectorized equivalent of calling _score_row() in a loop — same math,
        same result, but ~1000x faster on a large date range. The row-by-row
        version calls each of the ~247 ambient regression models once PER ROW
        (3M+ individual sklearn .predict() calls for a full 2-month range,
        which is what made /history/trend hang for ~10 minutes). This version
        calls each model ONCE on the whole batch instead.

        Returns a DataFrame indexed like `df`, with health_index/raw_score
        columns, NaN for rows that are missing required features (same rule
        as _score_row: not scoreable).
        """
        required = self.selected_features + ["Panel_Temp_mean_24h_calendar", "Panel_Humidity_mean_24h_calendar"]
        scoreable = df.dropna(subset=required)
        if len(scoreable) == 0:
            return pd.DataFrame({"health_index": [], "raw_anomaly_score": []})

        X_amb = self._build_ambient_design_batch(scoreable)
        resid = np.column_stack([
            scoreable[f].values - self.ambient["models"][f].predict(X_amb) for f in self.machine_features
        ])
        X_scaled = self.scaler.transform(resid)
        raw_scores = self.iso.decision_function(X_scaled)

        lo_raw, lo_hi = self._hi_score_min["example_lo"]
        hi_raw, hi_hi = self._hi_score_min["example_hi"]
        if hi_raw == lo_raw:
            health_index = np.full(len(raw_scores), 50.0)
        else:
            slope = (hi_hi - lo_hi) / (hi_raw - lo_raw)
            health_index = lo_hi + slope * (raw_scores - lo_raw)
        health_index = np.clip(health_index, 0, 100)

        return pd.DataFrame(
            {"health_index": health_index, "raw_anomaly_score": raw_scores}, index=scoreable.index
        )

    def _find_nearest_row(self, requested_timestamp):
        requested_timestamp = pd.Timestamp(requested_timestamp)
        idx = (self.features["timestamp"] - requested_timestamp).abs().idxmin()
        row = self.features.loc[idx]
        gap = abs((row["timestamp"] - requested_timestamp).total_seconds())
        return row, row["timestamp"], gap

    def _score_row(self, row):
        """Returns (health_index, raw_score) for an ACTIVE row with complete features, or None if not scoreable."""
        if any(pd.isna(row.get(c)) for c in self.selected_features):
            return None
        if pd.isna(row.get("Panel_Temp_mean_24h_calendar")) or pd.isna(row.get("Panel_Humidity_mean_24h_calendar")):
            return None
        X_amb = self._build_ambient_design(row)
        resid = np.array([[row[f] - self.ambient["models"][f].predict(X_amb)[0] for f in self.machine_features]])
        X_scaled = self.scaler.transform(resid)
        raw_score = float(self.iso.decision_function(X_scaled)[0])
        lo_raw, lo_hi = self._hi_score_min["example_lo"]
        hi_raw, hi_hi = self._hi_score_min["example_hi"]
        # linear map using the two calibration anchor points (equivalent to the training min/max scaling)
        if hi_raw == lo_raw:
            health_index = 50.0
        else:
            slope = (hi_hi - lo_hi) / (hi_raw - lo_raw)
            health_index = lo_hi + slope * (raw_score - lo_raw)
        health_index = float(np.clip(health_index, 0, 100))
        return health_index, raw_score

    def _risk_state(self, health_index):
        p25 = float(self.thresholds["states"]["NORMAL"]["condition"].split(">= ")[1])
        watch_lo = float(self.thresholds["states"]["WATCH"]["condition"].split(" <= ")[0])
        alert_lo = float(self.thresholds["states"]["ALERT"]["condition"].split(" <= ")[0])
        if health_index >= p25:
            return "NORMAL"
        if health_index >= watch_lo:
            return "WATCH"
        if health_index >= alert_lo:
            return "ALERT"
        return "CRITICAL"

    def _top_parameters_for_row(self, row, n=5):
        candidates = [p["feature"] for p in self.top_params_global]
        baseline_means = self.active_baseline[candidates].mean()
        baseline_stds = self.active_baseline[candidates].std().replace(0, 1e-6)
        importance_by_feat = {p["feature"]: p["importance"] for p in self.top_params_global}
        rows = []
        for feat in candidates:
            cur = row[feat]
            if pd.isna(cur):
                continue
            baseline = baseline_means[feat]
            deviation_pct = (cur - baseline) / (abs(baseline) + 1e-6) * 100
            z = (cur - baseline) / baseline_stds[feat]
            rows.append({
                "feature": feat, "subsystem": self.subsystem_map.get(feat.split("_mean_")[0].split("_std_")[0]
                                                                       .split("_min_")[0].split("_max_")[0]
                                                                       .split("_range_")[0].split("_roc_")[0],
                                                                       "unmapped"),
                "current": round(float(cur), 2), "baseline": round(float(baseline), 2),
                "deviation_pct": round(float(deviation_pct), 1), "z_score": round(float(z), 2),
                "global_importance": importance_by_feat[feat],
            })
        rows.sort(key=lambda r: -abs(r["z_score"]))
        return rows[:n]

    def _prognostics_as_of(self, as_of_timestamp):
        from scipy import stats
        hist = pd.concat([
            pd.read_csv(f"{MODEL_DIR}/health_index_v3_train.csv", parse_dates=["timestamp"]),
            pd.read_csv(f"{MODEL_DIR}/health_index_v3_val.csv", parse_dates=["timestamp"]),
            pd.read_csv(f"{MODEL_DIR}/health_index_v3_test.csv", parse_dates=["timestamp"]),
        ])
        hist = hist[hist["timestamp"] <= as_of_timestamp].sort_values("timestamp")
        daily = hist.set_index("timestamp")["health_index"].resample("1D").mean().dropna()
        if len(daily) < 10:
            return {"status": "NOT_AVAILABLE", "reason": "fewer than 10 days of history as of this timestamp"}
        x = np.arange(len(daily))
        slope, intercept, r, p, se = stats.linregress(x, daily.values)
        if slope >= 0 or p >= 0.05:
            return {"status": "NOT_AVAILABLE", "trend_slope_hi_per_day": round(float(slope), 4),
                    "p_value": round(float(p), 4),
                    "reason": "no statistically significant downward trend as of this timestamp"}

        # Statistical significance alone isn't enough to justify a specific date.
        # With ~55-60 daily points, p<0.05 is reachable even for a very weak,
        # noisy fit -- verified on this exact dataset: r^2 as low as 0.08-0.10
        # still crossed p<0.05 and would have extrapolated 300+ days from ~2
        # months of history. Two additional gates, both required:
        r_squared = r ** 2
        MIN_R_SQUARED = 0.30
        MAX_EXTRAPOLATION_DAYS = 90  # roughly 1.5x the available history span

        critical_threshold = float(self.thresholds["states"]["CRITICAL"]["condition"].split("< ")[1])
        current_hi = float(daily.iloc[-1])
        if current_hi <= critical_threshold:
            return {"status": "NOT_AVAILABLE", "trend_slope_hi_per_day": round(float(slope), 4),
                    "p_value": round(float(p), 4),
                    "reason": "current health index is already at or below the CRITICAL threshold; "
                              "a forward-looking horizon isn't meaningful here"}
        days_to_critical = (current_hi - critical_threshold) / (-slope)

        if r_squared < MIN_R_SQUARED:
            return {"status": "NOT_AVAILABLE", "trend_slope_hi_per_day": round(float(slope), 4),
                    "p_value": round(float(p), 4), "r_squared": round(float(r_squared), 3),
                    "reason": f"trend is statistically significant (p={p:.3f}) but too weak/noisy to act on "
                              f"(r^2={r_squared:.2f}, explains only {r_squared*100:.0f}% of variance) — "
                              f"not extrapolated into a specific date"}
        if days_to_critical > MAX_EXTRAPOLATION_DAYS:
            return {"status": "NOT_AVAILABLE", "trend_slope_hi_per_day": round(float(slope), 4),
                    "p_value": round(float(p), 4), "r_squared": round(float(r_squared), 3),
                    "reason": f"projected horizon ({days_to_critical:.0f} days) exceeds a defensible "
                              f"extrapolation range given only {len(daily)} days of history — not "
                              f"extrapolated into a specific date"}

        return {
            "status": "ESTIMATED_MAINTENANCE_HORIZON",
            "trend_slope_hi_per_day": round(float(slope), 4), "p_value": round(float(p), 4),
            "r_squared": round(float(r_squared), 3),
            "current_health_index": round(current_hi, 1), "critical_threshold": critical_threshold,
            "estimated_days_to_critical": round(float(days_to_critical), 2),
            "caution": "A statistically significant slope can also reflect a step-change event "
                       "(e.g. a service action) rather than continuous degradation — verified on this "
                       "exact dataset (see Hyd Oil Temp, 2026-05-19/20). Cross-check against maintenance "
                       "logs before treating this as a real degradation trend.",
        }

    # -----------------------------------------------------------------
    # Energy Intelligence (spec Parts 2-8). See energy_baseline.py for the
    # full derivation and its documented limitations (apparent power only,
    # no RPM/spindle-load signal, state-conditioned baseline).
    # -----------------------------------------------------------------
    def _compute_energy(self, row):
        """
        Computable for ANY operating state (only needs raw Stabilizer V/I,
        not the full rolling-feature set), unlike health scoring. Returns
        a dict; energy_status is NOT_APPLICABLE outside ACTIVE state.
        """
        v_cols = ["Stabilizer R Voltage", "Stabilizer Y Voltage", "Stabilizer B Voltage"]
        i_cols = ["Stabilizer R Current", "Stabilizer Y Current", "Stabilizer B Current"]
        if any(pd.isna(row.get(c)) for c in v_cols + i_cols):
            return {"actual_power_kva": None, "expected_power_kva": None, "power_deviation_kva": None,
                    "energy_excess_percent": None, "energy_status": "NOT_AVAILABLE",
                    "reason": "Stabilizer voltage/current reading missing at this timestamp"}

        actual = sum(row[v] * row[i] for v, i in zip(v_cols, i_cols)) / 1000
        state = row["operating_state"]
        baseline = self.energy_baseline.get(state)
        if baseline is None:
            return {"actual_power_kva": round(float(actual), 2), "expected_power_kva": None,
                    "power_deviation_kva": None, "energy_excess_percent": None, "energy_status": "NOT_APPLICABLE"}

        expected = baseline["expected_power_kva"]
        deviation = actual - expected
        pct = None if expected == 0 else (deviation / expected) * 100

        if state != "ACTIVE" or pct is None:
            status = "NOT_APPLICABLE"
        else:
            tp = self.energy_thresholds["thresholds_percent"]
            p75, p90, p97 = tp["p75"], tp["p90"], tp["p97"]
            if pct >= p97:
                status = "ENERGY_SAVING_OPPORTUNITY"
            elif pct >= p90:
                status = "ENERGY_ALERT"
            elif pct >= p75:
                status = "ENERGY_WATCH"
            else:
                status = "NORMAL_ENERGY"

        return {
            "actual_power_kva": round(float(actual), 2),
            "expected_power_kva": round(float(expected), 2),
            "power_deviation_kva": round(float(deviation), 2),
            "energy_excess_percent": round(float(pct), 1) if pct is not None else None,
            "energy_status": status,
            "power_unit_note": "apparent power (kVA) — no power factor available in this dataset, not true kW",
        }

    def _top_energy_driver_for_row(self, row, n=3):
        candidates = [c["feature"] for c in self.energy_drivers_global]
        baseline_means = self.active_baseline[candidates].mean()
        baseline_stds = self.active_baseline[candidates].std().replace(0, 1e-6)
        subsystem_by_feat = {c["feature"]: c["subsystem"] for c in self.energy_drivers_global}
        rows = []
        for feat in candidates:
            cur = row.get(feat)
            if pd.isna(cur):
                continue
            baseline = baseline_means[feat]
            z = (cur - baseline) / baseline_stds[feat]
            rows.append({"feature": feat, "subsystem": subsystem_by_feat[feat],
                         "current": round(float(cur), 2), "baseline": round(float(baseline), 2),
                         "z_score": round(float(z), 2)})
        rows.sort(key=lambda r: -abs(r["z_score"]))
        return rows[:n]

    @staticmethod
    def _joint_decision(risk_state, energy_status):
        """Same logic as joint_decision_engine.py — kept in sync manually since
        it's a handful of lines; see that file for the standalone/testable version."""
        if energy_status in ("NOT_APPLICABLE", "NOT_AVAILABLE") or risk_state is None:
            return {"joint_flag": "NOT_APPLICABLE",
                     "explanation": "Machine not in a state where both health and energy are scoreable."}
        health_degraded = risk_state in ("WATCH", "ALERT", "CRITICAL")
        energy_elevated = energy_status in ("ENERGY_WATCH", "ENERGY_ALERT", "ENERGY_SAVING_OPPORTUNITY")
        if health_degraded and energy_elevated:
            return {"joint_flag": "MAINTENANCE + ENERGY EFFICIENCY OPPORTUNITY",
                     "explanation": f"Health risk is {risk_state} AND energy status is {energy_status} at the same time."}
        if not health_degraded and energy_elevated:
            return {"joint_flag": "ENERGY OPTIMIZATION OPPORTUNITY",
                     "explanation": f"Health is {risk_state} (not degraded) but energy status is {energy_status}."}
        if health_degraded and not energy_elevated:
            return {"joint_flag": "MAINTENANCE RISK",
                     "explanation": f"Health risk is {risk_state} but energy consumption is normal."}
        return {"joint_flag": "NORMAL OPERATION", "explanation": "Both health and energy are within normal ranges."}

    def score_energy_batch(self, df):
        """
        Vectorized energy computation for a date range — same pattern as
        score_batch() for health, for the same reason: a per-row Python loop
        over a multi-week range is what caused the earlier /history/trend
        hang (see CHANGELOG). Works for any operating state, unlike
        score_batch() which is ACTIVE-only.
        """
        v_cols = ["Stabilizer R Voltage", "Stabilizer Y Voltage", "Stabilizer B Voltage"]
        i_cols = ["Stabilizer R Current", "Stabilizer Y Current", "Stabilizer B Current"]
        valid = df.dropna(subset=v_cols + i_cols).copy()
        if len(valid) == 0:
            return pd.DataFrame({"actual_power_kva": [], "expected_power_kva": [], "energy_status": []})

        actual = sum(valid[v] * valid[i] for v, i in zip(v_cols, i_cols)) / 1000
        expected = valid["operating_state"].map(lambda s: self.energy_baseline.get(s, {}).get("expected_power_kva"))
        deviation = actual - expected
        pct = np.where(expected.values == 0, np.nan, deviation.values / expected.values.astype(float) * 100)

        tp = self.energy_thresholds["thresholds_percent"]
        p75, p90, p97 = tp["p75"], tp["p90"], tp["p97"]

        is_active = (valid["operating_state"] == "ACTIVE").values
        status = np.where(~is_active | np.isnan(pct), "NOT_APPLICABLE",
                  np.where(pct >= p97, "ENERGY_SAVING_OPPORTUNITY",
                  np.where(pct >= p90, "ENERGY_ALERT",
                  np.where(pct >= p75, "ENERGY_WATCH", "NORMAL_ENERGY"))))

        return pd.DataFrame({
            "actual_power_kva": actual.round(2).values,
            "expected_power_kva": expected.astype(float).round(2).values,
            "power_deviation_kva": deviation.round(2).values,
            "energy_excess_percent": np.round(pct, 1),
            "energy_status": status,
        }, index=valid.index)

    def predict_historical(self, requested_timestamp):
        row, matched_ts, gap_seconds = self._find_nearest_row(requested_timestamp)
        operating_state = row["operating_state"]
        if pd.isna(operating_state):
            # Genuine sensor gap: the raw Stabilizer voltage/current reading
            # itself is missing at this timestamp (documented ~1.2% per-sensor
            # missingness from the raw data, see data_quality_report.md) --
            # not a bug, a real gap. Label it honestly rather than leaking NaN.
            operating_state = "NO_DATA"

        output = {
            "timestamp_requested": str(pd.Timestamp(requested_timestamp)),
            "timestamp_matched": str(matched_ts),
            "match_gap_seconds": gap_seconds,
            "mode": "historical",
            "machine_state": operating_state,
            "health_index": None, "risk_level": None, "anomaly_score": None,
            "failure_probability": None,  # never supported — no failure labels exist
            "rul": None,  # never claimed as validated RUL — see maintenance_horizon
            "maintenance_horizon": None,
            "dominant_subsystem": None, "top_parameters": [],
            "maintenance_category": None, "priority": None, "recommendation": None,
            "predicted_maintenance_datetime": None, "confidence": None,
            # --- energy fields (spec Parts 2-8) — additive, existing fields above unchanged ---
            "actual_power_kva": None, "expected_power_kva": None, "power_deviation_kva": None,
            "energy_excess_percent": None, "energy_status": None,
            "top_energy_drivers": [], "energy_recommendation": None,
            "joint_flag": None, "joint_explanation": None,
        }

        # Energy is computable for any operating state (only needs raw
        # Stabilizer readings), so compute it before the ACTIVE-only gate below.
        energy = self._compute_energy(row)
        output["actual_power_kva"] = energy.get("actual_power_kva")
        output["expected_power_kva"] = energy.get("expected_power_kva")
        output["power_deviation_kva"] = energy.get("power_deviation_kva")
        output["energy_excess_percent"] = energy.get("energy_excess_percent")
        output["energy_status"] = energy.get("energy_status")

        if operating_state != "ACTIVE":
            if operating_state == "NO_DATA":
                output["recommendation"] = ("No sensor reading available at this timestamp — this is a gap in "
                                             "the raw data (~1.2% of timestamps), not a machine state.")
            else:
                output["recommendation"] = "Machine not operating at this timestamp — no health assessment applicable."
            joint = self._joint_decision(None, energy.get("energy_status"))
            output["joint_flag"], output["joint_explanation"] = joint["joint_flag"], joint["explanation"]
            return output

        if energy.get("energy_status") in ("ENERGY_SAVING_OPPORTUNITY", "ENERGY_ALERT"):
            drivers = self._top_energy_driver_for_row(row)
            output["top_energy_drivers"] = drivers
            output["energy_recommendation"] = (
                f"Investigate {drivers[0]['subsystem']} — statistically associated with elevated power draw "
                f"at this timestamp (weak association, see energy_drivers_report.json for correlation strength)."
                if drivers else "Elevated energy use detected, but no candidate driver stood out from baseline."
            )
        elif energy.get("energy_status") == "ENERGY_WATCH":
            output["energy_recommendation"] = "Monitor — power draw above typical range but below saving-opportunity threshold."
        elif energy.get("energy_status") == "NORMAL_ENERGY":
            output["energy_recommendation"] = "No energy action needed."

        scored = self._score_row(row)
        if scored is None:
            output["recommendation"] = "INSUFFICIENT DATA — required features unavailable at this timestamp."
            joint = self._joint_decision(None, energy.get("energy_status"))
            output["joint_flag"], output["joint_explanation"] = joint["joint_flag"], joint["explanation"]
            return output

        health_index, raw_score = scored
        risk = self._risk_state(health_index)
        top_params = self._top_parameters_for_row(row)
        dominant = top_params[0]["subsystem"] if top_params else None
        prognostics = self._prognostics_as_of(matched_ts)

        # RUL and predicted maintenance date — ONLY populated when the trend
        # is statistically validated (see _prognostics_as_of). Never fabricated
        # for the common case where no significant trend exists: spec Part 11
        # requires "Maintenance horizon available but exact date is not
        # validated" instead of a made-up date in that case.
        rul_hours = None
        predicted_maintenance_datetime = None
        if prognostics.get("status") == "ESTIMATED_MAINTENANCE_HORIZON":
            rul_hours = round(prognostics["estimated_days_to_critical"] * 24, 1)
            predicted_maintenance_datetime = str(
                pd.Timestamp(matched_ts) + pd.Timedelta(days=prognostics["estimated_days_to_critical"])
            )

        windows = {"CRITICAL": "Immediate — within 24 hours", "ALERT": "Within 1 week",
                   "WATCH": "Monitor; next scheduled PM", "NORMAL": "No action needed"}
        recommendation = (f"Inspect {dominant} — dominant contributor to current anomaly score"
                           if risk in ("ALERT", "CRITICAL") and dominant
                           else "Continue normal monitoring" if risk == "WATCH" else "No action required")

        output.update({
            "health_index": round(health_index, 1),
            "anomaly_score": round(raw_score, 4),
            "risk_level": risk,
            "maintenance_horizon": prognostics,
            "rul": rul_hours,
            "rul_note": ("Estimated Remaining Time — MODEL ESTIMATE, not a validated RUL (no failure "
                         "labels exist in this dataset). See maintenance_horizon.caution."
                         if rul_hours is not None else
                         "Maintenance horizon available but exact date is not validated: "
                         + prognostics.get("reason", "no significant degradation trend detected")),
            "predicted_maintenance_datetime": predicted_maintenance_datetime,
            "dominant_subsystem": dominant,
            "top_parameters": top_params,
            "maintenance_category": ("Condition-based inspection" if risk in ("ALERT", "CRITICAL")
                                      else "Routine monitoring" if risk == "WATCH" else "None"),
            "priority": risk,
            "recommendation": recommendation,
            "maintenance_window": windows[risk],
            "confidence": "LOW" if prognostics.get("status") == "NOT_AVAILABLE" else "MEDIUM",
        })

        joint = self._joint_decision(risk, energy.get("energy_status"))
        output["joint_flag"], output["joint_explanation"] = joint["joint_flag"], joint["explanation"]

        return output


if __name__ == "__main__":
    engine = PredictionEngine()

    test_cases = [
        ("2026-04-23 00:05:00", "pre-service, well into a session"),
        ("2026-05-25 14:25:00", "post-service (after the oil-temp step change), well into a session"),
        ("2026-04-01 12:00:00", "known OFF period — should return not-operating"),
        ("2026-06-15 09:00:00", "outside the dataset entirely — nearest-match test"),
    ]
    for ts, desc in test_cases:
        result = engine.predict_historical(ts)
        print(f"\n=== {desc} ===")
        print(json.dumps(result, indent=2, default=str))
