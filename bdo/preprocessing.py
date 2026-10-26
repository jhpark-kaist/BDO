import numpy as np


class RobustScaler:
    """Normalization used for pre-training and downstream evaluation (paper, App. D.1).

    1. Drop the first `drop_initial` time steps of each scan (scanner warm-up).
    2. Subtract each ROI's temporal mean, separately for each scan.
    3. Subtract the per-ROI median and divide by the per-ROI interquartile range. Both are computed
       over all time steps of all scans passed to `fit`.

    In the paper, the statistics of each downstream dataset are fitted on its training split.
    """

    def __init__(self, drop_initial: int = 10):
        self.drop_initial = drop_initial
        self.median = None
        self.iqr = None

    def _center(self, scan: np.ndarray) -> np.ndarray:
        scan = np.asarray(scan, dtype=np.float64)[self.drop_initial:]
        return scan - scan.mean(axis=0)

    def fit(self, scans) -> "RobustScaler":
        """scans: iterable of [T_i, N] arrays."""
        pooled = np.concatenate([self._center(s) for s in scans], axis=0)
        self.median = np.median(pooled, axis=0)
        q25, q75 = np.nanpercentile(pooled, [25, 75], axis=0)
        self.iqr = np.where(q75 - q25 == 0, 1e-10, q75 - q25)
        return self

    def transform(self, scan: np.ndarray) -> np.ndarray:
        """[T, N] -> [T - drop_initial, N], float32."""
        if self.median is None:
            raise RuntimeError("RobustScaler is not fitted")
        return ((self._center(scan) - self.median) / self.iqr).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, drop_initial=self.drop_initial, median=self.median, iqr=self.iqr)

    @classmethod
    def load(cls, path: str) -> "RobustScaler":
        stats = np.load(path)
        scaler = cls(int(stats["drop_initial"]))
        scaler.median, scaler.iqr = stats["median"], stats["iqr"]
        return scaler


def regular_times(num_steps: int, tr: float) -> np.ndarray:
    """Time stamps in seconds of a scan with repetition time `tr`."""
    return np.arange(num_steps, dtype=np.float32) * np.float32(tr)
