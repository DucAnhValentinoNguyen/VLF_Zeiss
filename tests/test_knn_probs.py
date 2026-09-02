import numpy as np

from vlfz.eval.edl import evidential_knn, fit_evidence_scale, uncertainty_block
from vlfz.eval.knn import knn_vote


def _blobs(n=300, d=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, d)) * 5
    X, y = [], []
    for c in range(3):
        X.append(centers[c] + rng.normal(size=(n, d)))
        y += [c] * n
    return np.concatenate(X), np.array(y)


def test_knn_probs_valid_and_accurate():
    X, y = _blobs()
    ntr = len(y) * 2 // 3
    idx = np.random.default_rng(1).permutation(len(y))
    tr, te = idx[:ntr], idx[ntr:]
    out = knn_vote(X[tr], y[tr], X[te], n_classes=3, k=20, tau=0.1)
    p = out["probs"]
    assert p.shape == (len(te), 3)
    assert np.allclose(p.sum(1), 1.0, atol=1e-6)
    assert (p >= 0).all()
    acc = (p.argmax(1) == y[te]).mean()
    assert acc > 0.9, acc
    assert np.isfinite(out["logits"]).all()


def test_evidential_knn_dirichlet_props():
    X, y = _blobs()
    ntr = len(y) * 2 // 3
    idx = np.random.default_rng(2).permutation(len(y))
    tr, te, cal = idx[:ntr], idx[ntr : ntr + 100], idx[ntr + 100 :]
    kn = knn_vote(X[tr], y[tr], X[te], n_classes=3, k=20, tau=0.1)
    kn_cal = knn_vote(X[tr], y[tr], X[cal], n_classes=3, k=20, tau=0.1)

    ek = evidential_knn(kn["vote_mass"], scale=1.0)
    assert (ek["alpha"] > 1.0 - 1e-9).all()
    assert np.allclose(ek["prob"].sum(1), 1.0, atol=1e-6)
    assert ((ek["vacuity"] > 0) & (ek["vacuity"] <= 1)).all()

    lam = fit_evidence_scale(kn_cal["vote_mass"], y[cal])
    assert lam > 0
    ub = uncertainty_block(ek["prob"], ek["vacuity"], y[te])
    assert 0.0 <= ub["vacuity_mean"] <= 1.0
