import numpy as np
from learnloop.data import FEATURES, make_dataset
from learnloop.train import run

def test_dataset():
    X, y, groups = make_dataset(n=4000, seed=1)
    assert X.shape == (4000, len(FEATURES))
    assert set(np.unique(y)) <= {0, 1}
    assert 0.1 < y.mean() < 0.6
    assert set(np.unique(groups)) == {"working", "not_working"}

def test_model_has_signal_and_subgroup_report():
    report = run(seed=0)
    assert report["stronger_model"]["roc_auc"] >= 0.74, report
    assert report["stronger_model"]["recall_at_0_5"] >= 0.3
    assert report["subgroup_diagnostic"]["attribute"] == "work_status"
