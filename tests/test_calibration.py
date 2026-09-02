import numpy as np

from vlfz.eval.calibration import (
    calibration_report,
    ece_equal_width,
    fit_temperature,
    nll,
)


def _synth_calibrated(n=6000, c=3, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, c)) * 1.5
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(c, p=pi) for pi in p])   # labels drawn from p -> calibrated
    return logits, p, y


def test_calibrated_has_low_ece():
    _, p, y = _synth_calibrated()
    ece, _ = ece_equal_width(p, y, 15)
    assert ece < 0.03, ece


def test_overconfident_temperature_gt_one_and_helps():
    logits, _, y = _synth_calibrated()
    sharp = logits * 4.0                       # overconfident
    n = len(y)
    cal, q = slice(0, n // 2), slice(n // 2, n)
    T = fit_temperature(sharp[cal], y[cal])
    assert T > 1.2, T
    pre = nll(sharp[q], y[q])
    post = nll(sharp[q] / T, y[q])
    assert post < pre, (pre, post)


def test_nll_finite_on_onehot_wrong():
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = np.array([1, 0])                       # both wrong, zero prob on truth
    assert np.isfinite(nll(p, y))


def test_matches_torchmetrics():
    from torchmetrics.classification import MulticlassCalibrationError
    import torch

    _, p, y = _synth_calibrated(seed=3)
    ours, _ = ece_equal_width(p, y, 15)
    tm = MulticlassCalibrationError(num_classes=p.shape[1], n_bins=15, norm="l1")
    ref = float(tm(torch.tensor(p), torch.tensor(y)))
    assert abs(ours - ref) < 1e-6, (ours, ref)


def test_report_structure():
    logits, _, y = _synth_calibrated()
    n = len(y)
    rep = calibration_report(logits[n // 2:], y[n // 2:], logits[: n // 2], y[: n // 2])
    assert set(rep) == {"pre", "post", "T"}
    for k in ("ece_ew", "ece_adaptive", "nll", "brier"):
        assert np.isfinite(rep["pre"][k]) and np.isfinite(rep["post"][k])
    assert rep["T"] > 0
