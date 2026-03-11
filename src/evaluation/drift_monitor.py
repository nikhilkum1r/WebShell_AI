"""
Drift Monitor — Simple MLOps utility for tracking model confidence.
Logs a warning to retrain_live.log if rolling confidence drops below threshold.
"""

import logging
from collections import deque
from pathlib import Path

# Configure monitoring logger
log_path = Path("retrain_live.log")
monitor_logger = logging.getLogger("drift_monitor")
monitor_logger.setLevel(logging.INFO)
# Ensure file handler exists
fh = logging.FileHandler(log_path)
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
monitor_logger.addHandler(fh)

class DriftMonitor:
    def __init__(self, window_size: int = 50, threshold: float = 0.7):
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)
    
    def add_prediction(self, confidence: float):
        """Adds a prediction confidence score and checks for drift."""
        self.history.append(confidence)
        
        if len(self.history) >= self.window_size:
            avg_conf = sum(self.history) / len(self.history)
            if avg_conf < self.threshold:
                monitor_logger.warning(
                    f"POTENTIAL CONCEPT DRIFT DETECTED: Rolling average confidence ({avg_conf:.2f}) "
                    f"dropped below threshold ({self.threshold}). Retraining recommended."
                )
            elif len(self.history) % 10 == 0:
                monitor_logger.info(f"Monitoring: Rolling average confidence is {avg_conf:.2f}")

# Singleton instance
_monitor = None

def get_monitor() -> DriftMonitor:
    global _monitor
    if _monitor is None:
        _monitor = DriftMonitor()
    return _monitor
