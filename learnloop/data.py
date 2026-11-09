"""Deterministic, domain-specific synthetic dataset for LearnLoop-Predict — Student Support Research Prototype."""

from __future__ import annotations
import numpy as np

FEATURES = ['attendance', 'score', 'gpa', 'hours', 'logins', 'late', 'working', 'aid']
GROUP_NAME = 'work_status'
GROUP_SPLIT = False

def make_dataset(n=2400, seed=42):
    if n < 80:
        raise ValueError("n must be at least 80 for a useful holdout")
    rng = np.random.default_rng(seed)
    attendance = np.clip(rng.normal(82, 14, n), 20, 100)
    score = np.clip(rng.normal(72, 16, n), 0, 100)
    gpa = np.clip(rng.normal(2.9, .7, n), 0, 4)
    hours = np.clip(rng.normal(9, 5, n), 0, 40)
    logins = np.clip(rng.poisson(10, n), 0, 60).astype(float)
    late = rng.poisson(1.2, n).astype(float)
    working = rng.integers(0, 2, n).astype(float)
    aid = rng.integers(0, 2, n).astype(float)
    z = 9.3 - .045*attendance - .040*score - 1.10*gpa - .070*hours - .045*logins + .40*late + .35*working - .10*aid
    p = 1/(1+np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    groups = np.where(working > 0, "working", "not_working")
    X = np.column_stack([attendance, score, gpa, hours, logins, late, working, aid]).astype(float)
    if X.shape != (n, len(FEATURES)) or not np.isfinite(X).all():
        raise ValueError("generator produced invalid feature matrix")
    if np.asarray(groups).shape != (n,):
        raise ValueError("generator produced invalid group vector")
    return X, y.astype(int), np.asarray(groups)
