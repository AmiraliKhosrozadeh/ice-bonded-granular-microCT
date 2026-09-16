"""One definition of strain and of the bead-to-bead assignment.

Both were re-implemented in four scripts and both were wrong in the same way
everywhere, so they live here now.

WHY GREEN-LAGRANGE AND NOT 0.5*(G+G^T)
The old code used the infinitesimal measure E = 0.5*(G+G^T).  That is only a
strain measure when displacement gradients and rotations are small, and here
they are not: measured over the thirteen cumulative fields, median ||G||_F runs
0.17-0.78, the median local rotation from the polar decomposition of F = I+G
runs 3.5-19.6 degrees, the 95th percentile reaches 22-66 degrees, and 1.6-16.9%
of beads exceed 30 degrees.  At that size the neglected quadratic term is about
30% of the retained linear part, and worse, the linear measure is not
frame-indifferent -- a bead that merely ROTATES registers a spurious strain.
Green-Lagrange, E = 0.5*(F^T F - I), is exact for arbitrary rotation and is the
standard Lagrangian measure for a field evaluated on the reference (scan-1)
configuration, which is exactly what this is.

WHY THE ASSIGNMENT MUST BE INJECTIVE
A plain cKDTree(Xk).query(pred, k=1) resolves every query independently, so two
scan-1 beads may claim the SAME scan-k bead and both pass a distance gate --
they then inherit a bitwise-identical displacement, which is physically
impossible (one bead cannot be in two places).  Measured on the unconstrained
version: Glass_100_1700_T7 3->4 accepted 350 matches onto only 254 distinct
targets, Glass_75_1700_T5_HR 2->3 accepted 340 onto 267.  The printed "98%
matched" was counting beads that found *a* bead, not beads that found *their
own*.  Greedy best-first assignment fixes it: closest pair wins, each target is
consumed once, and a bead left without a free target within tolerance is
dropped rather than given someone else's displacement.
"""
import numpy as np
from scipy.spatial import cKDTree


def assign_injective(pred, target, tol, k=6):
    """Greedy one-to-one match of predicted positions to target positions.

    Returns (src_idx, tgt_idx) into `pred` and `target`, closest pairs first,
    each target used at most once and every pair within `tol`.
    """
    if len(target) == 0 or len(pred) == 0:
        return np.empty(0, int), np.empty(0, int)
    kk = min(k, len(target))
    d, j = cKDTree(target).query(pred, k=kk, distance_upper_bound=tol)
    d = np.atleast_2d(d.T).T if kk > 1 else d[:, None]
    j = np.atleast_2d(j.T).T if kk > 1 else j[:, None]
    src = np.repeat(np.arange(len(pred)), kk)
    tgt, dist = j.ravel(), d.ravel()
    ok = np.isfinite(dist) & (tgt < len(target))
    src, tgt, dist = src[ok], tgt[ok], dist[ok]
    order = np.argsort(dist, kind='stable')          # closest pair wins
    used_s = np.zeros(len(pred), bool)
    used_t = np.zeros(len(target), bool)
    S, T = [], []
    for i in order:
        s, t = src[i], tgt[i]
        if used_s[s] or used_t[t]:
            continue
        used_s[s] = used_t[t] = True
        S.append(s); T.append(t)
    o = np.argsort(S, kind='stable')
    return np.asarray(S, int)[o], np.asarray(T, int)[o]


def grad_u(X, U, k=12, intercept=True):
    """Least-squares displacement gradient over the k nearest neighbours.

    THE FIT MUST HAVE AN INTERCEPT.  Solving dU = G dX with no constant term
    forces the plane through the centre bead, so that bead's OWN non-affine
    motion is charged to the gradient.  The leverage of that is zero when the
    neighbours surround the bead and large when they do not -- which is exactly
    the free surface.  The result was a strain field elevated in the outermost
    bead layer and flat everywhere inside it.

    This was diagnosed with a null control rather than by argument.  On the
    real bead positions, including the real free surface, a SYNTHETIC field of
    known uniform strain is recovered exactly either way (max error 7e-16,
    perimeter/interior 1.0000), so one-sided extrapolation is NOT the problem.
    But feeding in the measured non-affine fluctuation with no radial structure
    at all reproduces the observed perimeter excess (1.171 +/- 0.027 against
    1.189 as shipped) -- the excess is manufactured from noise by the missing
    intercept.  Allowing the intercept drops it to 0.954.

    Solving for the constant is also strictly better on real gradients: on a
    field with an analytically known perimeter excess of 1.92 the intercept fit
    recovers 1.73 against the no-intercept 1.60, and its answer is stable in k
    (0.92/0.92/0.90/0.90 at k=8/12/20/30) where the old one drifts
    (1.11/1.22/1.29/1.51).

    intercept=False reproduces the old estimator, for comparison only.
    """
    tr = cKDTree(X)
    G = np.zeros((len(X), 3, 3))
    kk = min(k + 1, len(X))
    for i in range(len(X)):
        _, ii = tr.query(X[i], k=kk)
        ii = np.asarray(ii)[1:]
        dX, dU = X[ii] - X[i], U[ii] - U[i]
        if intercept:
            A = np.hstack([dX, np.ones((len(dX), 1))])
            sol, *_ = np.linalg.lstsq(A, dU, rcond=None)
            G[i] = sol[:3].T          # drop the constant, keep the gradient
        else:
            sol, *_ = np.linalg.lstsq(dX, dU, rcond=None)
            G[i] = sol.T
    return G


def _eq(E):
    v = np.trace(E, axis1=1, axis2=2)
    dv = E - (v[:, None, None] / 3.0) * np.eye(3)
    return np.sqrt(2.0 / 3.0 * (dv ** 2).sum(axis=(1, 2)))


def eq_strain(X, U, k=12, measure='green', intercept=True):
    """Equivalent strain. 'green' = Green-Lagrange (default), 'linear' = old."""
    G = grad_u(X, U, k, intercept)
    if measure == 'linear':
        return _eq(0.5 * (G + G.transpose(0, 2, 1)))
    F = G + np.eye(3)
    return _eq(0.5 * (F.transpose(0, 2, 1) @ F - np.eye(3)))


def rotation_deg(X, U, k=12):
    """Median local rigid rotation, for reporting how finite the field is."""
    F = grad_u(X, U, k) + np.eye(3)
    ang = np.zeros(len(F))
    for i, f in enumerate(F):
        u, _, vt = np.linalg.svd(f)
        Rm = u @ vt
        if np.linalg.det(Rm) < 0:
            u[:, -1] *= -1
            Rm = u @ vt
        ang[i] = np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
    return ang


def rigid_out(X, U):
    """Remove the least-squares rigid translation and rotation."""
    Xc = X - X.mean(axis=0)
    Rr = U - U.mean(axis=0)
    A = np.zeros((3 * len(X), 3))
    for i, r in enumerate(Xc):
        A[3*i:3*i+3] = [[0, r[2], -r[1]], [-r[2], 0, r[0]], [r[1], -r[0], 0]]
    om, *_ = np.linalg.lstsq(A, Rr.reshape(-1), rcond=None)
    return Rr - np.cross(np.tile(om, (len(X), 1)), Xc)
