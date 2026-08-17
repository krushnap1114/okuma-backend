# lstm_prediction_engine.py
#
# A completely independent second model: an LSTM Autoencoder trained fresh
# from the raw CBM export files (33 .xlsx files), built with no code or
# logic reused from prediction_engine.py's Isolation Forest pipeline.
#
# What this model does differently: it scores a 1-hour SEQUENCE (12 x 5-min
# readings) as a whole via reconstruction error, so it can catch anomalies
# that show up as an unusual temporal pattern across the hour -- not just an
# unusual single reading. The Isolation Forest model scores each 5-minute
# snapshot independently and cannot see this.
#
# What this model does NOT change: it still cannot produce a real predicted
# maintenance date or RUL. That requires failure/maintenance event labels,
# which do not exist anywhere in the raw source data (verified directly --
# see chat) -- a fact that is independent of which anomaly-detection
# architecture is used. This model's job is anomaly detection quality, not
# manufacturing a label that was never recorded.
import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib

BASE_DIR = Path(__file__).parent
WINDOW = 12


class LSTMEngine:
    def __init__(self, model_dir=BASE_DIR):
        model_dir = Path(model_dir)
        self.calibration = json.load(open(model_dir / "calibration.json"))
        self.feature_cols = json.load(open(model_dir / "feature_cols.json"))
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        # ONNX Runtime, not torch, for inference -- ~225KB model + a light
        # runtime, instead of a 1.2GB torch install, to keep the production
        # deployment fast and reliable. Exported from the trained PyTorch
        # model and verified to match it within float32 precision (~1e-6)
        # on real validation windows before being trusted here.
        self.session = ort.InferenceSession(str(model_dir / "lstm_autoencoder.onnx"))

        # Full cleaned 5-min-grid dataset, held in memory for windowed lookups
        self.df = pd.read_csv(model_dir / "wide_clean.csv", index_col=0, parse_dates=True)

    def _risk_state(self, hi):
        t = self.calibration["thresholds"]
        if hi is None:
            return None
        if hi >= t["normal"]:
            return "NORMAL"
        if hi >= t["watch"]:
            return "WATCH"
        if hi >= t["alert"]:
            return "ALERT"
        return "CRITICAL"

    def _error_to_hi(self, err):
        p50, p97 = self.calibration["p50"], self.calibration["p97"]
        ceiling = p97 * 1.5
        hi = 100 * (1 - (err - p50) / (ceiling - p50))
        return float(np.clip(hi, 0, 100))

    def score_at(self, timestamp):
        """Score the 1-hour window ending at `timestamp`. Returns a dict
        matching the shape the dashboard already expects (health_index,
        risk_level, machine_state, recommendation), so it can be wired into
        the existing frontend components without changing their contract --
        while the actual computation underneath is entirely independent."""
        ts = pd.Timestamp(timestamp)
        if ts not in self.df.index:
            # snap to nearest available 5-min grid point
            idx = self.df.index.get_indexer([ts], method="nearest")[0]
            ts = self.df.index[idx]

        loc = self.df.index.get_loc(ts)
        state = self.df["operating_state"].iloc[loc]

        if state != "ACTIVE":
            return {
                "timestamp_requested": str(timestamp),
                "timestamp_matched": str(ts),
                "machine_state": state,
                "health_index": None,
                "risk_level": None,
                "reconstruction_error": None,
                "recommendation": f"Machine not operating at this timestamp ({state}) -- no health assessment applicable.",
                "model": "lstm_autoencoder_v1",
            }

        if loc < WINDOW - 1:
            return {
                "timestamp_requested": str(timestamp), "timestamp_matched": str(ts),
                "machine_state": state, "health_index": None, "risk_level": None,
                "reconstruction_error": None,
                "recommendation": "INSUFFICIENT DATA -- fewer than 12 trailing readings (1 hour) available at this timestamp.",
                "model": "lstm_autoencoder_v1",
            }

        window_states = self.df["operating_state"].iloc[loc - WINDOW + 1: loc + 1]
        window_ts = self.df.index[loc - WINDOW + 1: loc + 1]
        contiguous = (np.diff(window_ts.values).astype("timedelta64[m]").astype(int) == 5).all()
        if not (window_states == "ACTIVE").all() or not contiguous:
            return {
                "timestamp_requested": str(timestamp), "timestamp_matched": str(ts),
                "machine_state": state, "health_index": None, "risk_level": None,
                "reconstruction_error": None,
                "recommendation": "INSUFFICIENT DATA -- the trailing 1-hour window includes a non-ACTIVE or non-contiguous period.",
                "model": "lstm_autoencoder_v1",
            }

        raw_window = self.df[self.feature_cols].iloc[loc - WINDOW + 1: loc + 1].to_numpy()
        scaled = self.scaler.transform(raw_window).astype(np.float32)
        x = scaled[np.newaxis, :, :]  # (1, WINDOW, n_features)
        recon = self.session.run(None, {"input": x})[0][0]
        err = float(((recon - scaled) ** 2).mean())
        hi = self._error_to_hi(err)
        risk = self._risk_state(hi)

        recs = {
            "NORMAL": "Sequence pattern over the last hour matches typical active operation. No action needed.",
            "WATCH": "Sequence pattern over the last hour deviates somewhat from typical operation. Monitor.",
            "ALERT": "Sequence pattern over the last hour deviates significantly from typical operation. Inspect soon.",
            "CRITICAL": "Sequence pattern over the last hour is highly atypical. Inspect promptly.",
        }

        return {
            "timestamp_requested": str(timestamp),
            "timestamp_matched": str(ts),
            "machine_state": state,
            "health_index": round(hi, 1),
            "risk_level": risk,
            "reconstruction_error": round(err, 4),
            "recommendation": recs[risk],
            "model": "lstm_autoencoder_v1",
            "window_hours": 1,
        }

    def trend(self, start=None, end=None):
        """Fast batched scoring for a trend chart: builds every valid
        (contiguous, 1-hour, ACTIVE) window in range and scores them all in
        one ONNX Runtime call, instead of looping score_at() -- ~15s looped
        vs sub-second batched for the full dataset."""
        df = self.df
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        values = df[self.feature_cols].to_numpy()
        timestamps = df.index.to_numpy()
        is_active = (df["operating_state"] == "ACTIVE").to_numpy()

        valid_end_idx = []
        for i in range(WINDOW - 1, len(df)):
            sl = slice(i - WINDOW + 1, i + 1)
            if not is_active[sl].all():
                continue
            diffs = np.diff(timestamps[sl]).astype("timedelta64[m]").astype(int)
            if not (diffs == 5).all():
                continue
            valid_end_idx.append(i)

        if not valid_end_idx:
            return []

        valid_end_idx = np.array(valid_end_idx)
        scaled_all = self.scaler.transform(values).astype(np.float32)
        batch = np.stack([scaled_all[i - WINDOW + 1:i + 1] for i in valid_end_idx])
        recon_batch = self.session.run(None, {"input": batch})[0]
        errs = ((recon_batch - batch) ** 2).mean(axis=(1, 2))
        his = [self._error_to_hi(e) for e in errs]

        return [
            {"timestamp": str(timestamps[i]), "health_index": round(h, 1)}
            for i, h in zip(valid_end_idx, his)
        ]
