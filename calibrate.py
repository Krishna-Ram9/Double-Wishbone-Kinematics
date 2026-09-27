"""
calibrate.py  (v2)
==================
Search double-wishbone hardpoints that hit target static roll-centre
height, camber-gain rate and roll-centre migration spread, subject to
realistic arm-length / ball-joint-offset constraints.

Outer loop = hardpoint search. Inner loop = exact 4-bar kinematics.

WHAT CHANGED FROM v1
--------------------
1.  least_squares (trust-region) replaces Nelder-Mead. The objective was
    already a sum of squares, so v1 was throwing away the structure that
    makes this problem easy. Converges in far fewer function evaluations.
2.  Constraints are residual entries, not a squared penalty folded into a
    scalar. Same effect, but the solver can see each violation separately.
3.  Bare `except Exception: return 1e6` is gone. It turned every
    unsolvable geometry into the same cliff, which Nelder-Mead handles
    badly and which silently hid real bugs. Unsolvable geometries now
    return a large but SMOOTH residual that still points downhill.
4.  Camber target is signed. v1 targeted the mean of absolute values, so
    a geometry gaining positive camber in bump scored identically to one
    gaining negative camber. Targets below are negative, i.e. negative
    camber in bump.
5.  The clearance term uses the corrected rim-clearance model in
    fourbar_solver v2.
6.  Results are written next to this file, not to a hardcoded
    /home/claude/suspension/ path that does not exist.
7.  half_track values match the report (625 front / 600 rear). v1 used
    600/590, so its output could not have produced the report's
    hardpoints even if it had been wired up.

Run:
    python3 calibrate.py
"""

import os

import numpy as np
from scipy.optimize import least_squares

from fourbar_solver import Corner, LinkageError

WHEEL_RADIUS = 540.0 / 2       # 540mm tyre OD (MRF ZTD1 200/540-13) -> 270 mm
RIM_RADIUS = (13 * 25.4) / 2   # 13 in rim -> 165.1 mm (diameter/2 proxy)
TRAVEL = 25.0               # +/- mm

HERE = os.path.dirname(os.path.abspath(__file__))

# residual weights: (roll centre, camber gain, migration spread)
WEIGHTS = (1.0, 8.0, 0.15)
CLEARANCE_MARGIN = 5.0      # mm of rim clearance to keep in hand


def build_corner(name, x, half_track, wc_height):
    """x = [LI_y, LI_z, LO_y, LO_z, UI_y, UI_z, UO_y, UO_z]."""
    return Corner(name,
                  LI=(x[0], x[1]), LO=(x[2], x[3]),
                  UI=(x[4], x[5]), UO=(x[6], x[7]),
                  wheel_center0=(half_track, wc_height),
                  wheel_radius=WHEEL_RADIUS, rim_radius=RIM_RADIUS)


def constraint_residuals(x, half_track):
    """One residual per constraint. Zero when satisfied, linear in the
    violation otherwise -- so the solver gets a usable gradient."""
    LI_y, LI_z, LO_y, LO_z, UI_y, UI_z, UO_y, UO_z = x
    L_lower = np.hypot(LO_y - LI_y, LO_z - LI_z)
    L_upper = np.hypot(UO_y - UI_y, UO_z - UI_z)
    L_upright = np.hypot(UO_y - LO_y, UO_z - LO_z)

    def band(v, lo, hi):
        return max(lo - v, 0.0) + max(v - hi, 0.0)

    k = 0.2   # constraint weight relative to the performance residuals
    return k * np.array([
        # ball joints near the wheel-centre plane (small scrub / kingpin offset)
        max(abs(LO_y - half_track) - 60.0, 0.0),
        max(abs(UO_y - half_track) - 60.0, 0.0),
        # inboard pivots meaningfully inboard of the ball joints
        band(LO_y - LI_y, 150.0, 550.0),
        band(UO_y - UI_y, 120.0, 550.0),
        # arm lengths inside a realistic FSAE envelope
        band(L_lower, 200.0, 420.0),
        band(L_upper, 180.0, 380.0),
        band(L_upright, 200.0, 380.0),
        # upper pivots above lower pivots
        max(LI_z - UI_z + 40.0, 0.0),
        max(LO_z - UO_z + 40.0, 0.0),
    ])


def residuals(x, half_track, wc_height, target_rc, target_camber_gain,
              target_spread):
    cons = constraint_residuals(x, half_track)
    big = np.array([1e3, 1e3, 1e3])
    try:
        c = build_corner("tmp", x, half_track, wc_height)
        p0 = c.static()
        rc0 = p0.roll_center_z
        if not np.isfinite(rc0):
            return np.concatenate([big, cons])
        cg = c.camber_gain_per_25mm(dz=TRAVEL, signed=True)
        dzs, rc, cam, clr = c.sweep(-TRAVEL, TRAVEL, 9)
        if not np.all(np.isfinite(rc)):
            return np.concatenate([big, cons])
        spread = float(np.nanmax(rc) - np.nanmin(rc))
    except LinkageError:
        # unsolvable geometry: large residual, but the constraint block
        # still points the solver back toward an assemblable linkage
        return np.concatenate([big, cons])

    clr_pen = np.clip(CLEARANCE_MARGIN - clr, 0.0, None)

    perf = np.array([
        WEIGHTS[0] * (rc0 - target_rc),
        WEIGHTS[1] * (cg - target_camber_gain),
        WEIGHTS[2] * (spread - target_spread),
    ])
    return np.concatenate([perf, 0.1 * clr_pen, cons])


def calibrate(name, half_track, wc_height, target_rc, target_camber_gain,
              target_spread, x0, seed=0, trials=3):
    best = None
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float)
    for trial in range(trials):
        start = x0 if trial == 0 else x0 * (1 + rng.normal(0, 0.02, x0.size))
        res = least_squares(
            residuals, start,
            args=(half_track, wc_height, target_rc, target_camber_gain,
                  target_spread),
            method="trf", x_scale="jac", diff_step=1e-4,
            xtol=1e-10, ftol=1e-12, max_nfev=4000,
        )
        if best is None or res.cost < best.cost:
            best = res
    return best


def describe(name, x, half_track, wc_height):
    c = build_corner(name, x, half_track, wc_height)
    p0 = c.static()
    dzs, rc, cam, clr = c.sweep(-TRAVEL, TRAVEL, 41)
    print(f"  {name}: RC {p0.roll_center_z:6.2f} mm | "
          f"camber {c.camber_gain_per_25mm(signed=True):+6.2f} deg/25mm | "
          f"spread {np.nanmax(rc) - np.nanmin(rc):6.2f} mm | "
          f"FVSA {p0.fvsa:7.1f} mm | min clearance {np.nanmin(clr):5.1f} mm")
    print(f"        arms: lower {c.L_lower:.1f}, upper {c.L_upper:.1f}, "
          f"upright {c.L_upright:.1f} mm")


if __name__ == "__main__":
    # RE-DERIVED FOR THE 13" PACKAGE (was: front RC 44.1 / cg -1.22 /
    # spread 32.0, rear RC 81.9 / cg -1.28 / spread 10.3 -- those were set
    # against the old 18" wheel radius; wc_height feeds straight into the
    # geometry, so a taller/shorter tyre shifts what RC is even reachable,
    # and those numbers no longer describe an achievable or meaningful
    # goal for this WHEEL_RADIUS).
    #
    # Method: anchored to the actual validated 13" CAD baseline in
    # double_wishbone_report.py (front RC 16.9 mm / cg -1.08 / spread
    # 42.8 mm, rear RC 26.4 mm / cg -0.93 / spread 9.4 mm -- reproduced
    # by running those hardpoints through this same solver). New targets
    # keep the old design intent -- rear RC held above front RC, negative
    # (top-in) camber gain both ends, tighter migration spread than
    # today's baseline -- as a modest, achievable improvement over that
    # baseline rather than a carried-over 18" number. Both targets below
    # were confirmed to converge to ~0 residual (i.e. reachable within
    # the arm-length/offset constraint envelope) before being committed
    # here. Re-check against updated CAD if the hardpoint envelope in
    # constraint_residuals() changes.
    # ---- FRONT ----
    half_track_f = 625.0
    x0_f = np.array([230, 120, 650, 60, 240, 300, 630, 340], dtype=float)
    res_f = calibrate("front", half_track_f, WHEEL_RADIUS,
                      target_rc=20.0, target_camber_gain=-1.10,
                      target_spread=30.0, x0=x0_f, seed=1)

    # ---- REAR ----
    half_track_r = 600.0
    x0_r = np.array([225, 140, 640, 55, 235, 290, 615, 335], dtype=float)
    res_r = calibrate("rear", half_track_r, WHEEL_RADIUS,
                      target_rc=32.0, target_camber_gain=-1.05,
                      target_spread=8.0, x0=x0_r, seed=2)

    print("\nCALIBRATED GEOMETRY [LI_y, LI_z, LO_y, LO_z, UI_y, UI_z, UO_y, UO_z]")
    print("  front:", np.round(res_f.x, 2))
    print("  rear :", np.round(res_r.x, 2))
    print(f"\n  front cost {res_f.cost:.4g} in {res_f.nfev} evals")
    print(f"  rear  cost {res_r.cost:.4g} in {res_r.nfev} evals")

    print("\nACHIEVED vs TARGET (front RC 20.0 / cg -1.10 / spread 30.0;"
          " rear RC 32.0 / cg -1.05 / spread 8.0)")
    describe("front", res_f.x, half_track_f, WHEEL_RADIUS)
    describe("rear", res_r.x, half_track_r, WHEEL_RADIUS)

    np.save(os.path.join(HERE, "front_x.npy"), res_f.x)
    np.save(os.path.join(HERE, "rear_x.npy"), res_r.x)
    print(f"\nSaved front_x.npy / rear_x.npy to {HERE}")
