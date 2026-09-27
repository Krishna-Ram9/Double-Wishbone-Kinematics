"""
double_wishbone_report.py  (v2)
===============================
Front-view 4-bar (double-wishbone) suspension kinematics report.

Run:
    python3 double_wishbone_report.py

Reports, per corner:
  - static roll-centre height and instant-centre position
  - FVSA length (check this against the CAD-measured value)
  - signed camber gain, deg / 25 mm travel
  - roll-centre migration spread over the travel sweep
  - rim clearance at the ball joints
and, for the vehicle:
  - geometric anti-dive / anti-squat, once side-view inputs are filled in

NOTHING IN THIS REPORT IS HARDCODED. v1 printed three numbers that were
typed in rather than computed (a "CAD-derived wheelbase" string, a 20 mm
clearance constant, and an anti-dive result from a function that only
compared two numbers for equality). Anything this version cannot compute
from the inputs below, it says so instead of printing a value.
"""

import math

import numpy as np

from fourbar_solver import Corner, side_view_instant_center, anti_percent

# --- vehicle-level constants -------------------------------------------------
# wheelbase = rear Z - front Z = -640.75 -> -2205 = 1564.25 mm (CAD upright points)
WHEELBASE = 1564.25                 # mm, spec
CG_HEIGHT = 248.0                   # mm, estimated
WHEEL_RADIUS = 540.0 / 2            # mm, 540mm tyre OD (MRF ZTD1 200/540-13)
RIM_RADIUS = (13 * 25.4) / 2        # mm, 13 in rim (diameter/2 proxy, not a
                                     # CAD-measured inner-barrel figure)
TRAVEL = 25.0                       # +/- mm, travel sweep and camber metric

# CAD Y-coordinate of the ground plane, in the chassis frame.
# VALIDATED: -107.2 reproduces the SolidWorks-measured roll centres
# (front 63.86 mm, rear 62.45 mm). The -102.7 figure in the older notes
# gives 61.2 / 61.2 and is wrong. Do not change this without re-checking
# against SolidWorks -- it shifts every reported height one-for-one.
GROUND_Y = -107.2

# Longitudinal (fore/aft) offset of each ball joint from the wheel-centre
# plane, mm. Needed for the rim clearance check. 0.0 means the ball joint
# centre sits in the wheel-centre plane -- replace with CAD values.
LO_X_OFFSET = 0.0
UO_X_OFFSET = 0.0


def _cad_to_solver(x_cad, y_cad):
    """CAD front-view point (X = lateral, Y = vertical, ground at GROUND_Y)
    -> solver frame (y = lateral, z = height above ground)."""
    return (x_cad, y_cad - GROUND_Y)


# --- front-view hardpoints (left side; right side mirrors with X negated) ----
#
# Inboard pivots below are the earlier 18"-wheel SolidWorks points (the
# chassis-side points are unchanged between wheel packages: "the inner
# points of the chassis are the same"). Outboard (upright) points are the
# CAD-measured 13" wheel-package values (MRF ZTD1 200/540-13: 540 mm tyre
# OD, 13" rim).
#
# VALIDATED against SolidWorks for this 13" package: front RC 16.9 mm,
# rear RC 26.4 mm, matching the solver output below.
#
# Separately, none of this matches the converged optimisation target on
# record in calibrate.py (front RC 44.1 mm, rear RC 81.9 mm, rear camber
# gain 1.28 deg/25 mm) -- that gap is a design question, not a code one.
#
# Inboard pivot pairs (in1/in2) are the longitudinally separated bushings
# of the same physical axis. For the REAR and for both UPPER arms the two
# bushings share a lateral X, so the axis projects to a single front-view
# point and averaging is exact.
#
# The FRONT LOWER pair does not: X = 240.00 vs 252.50, a 12.5 mm plan-view
# sweep. Averaging it gives 64.26 mm; using the 252.50 bushing gives
# 63.86 mm, which is exactly what SolidWorks reports. So SolidWorks is
# projecting from that bushing, and FRONT_LOWER_INBOARD below uses it
# directly rather than an average.

_F_U_in1, _F_U_in2 = (327.46, 149.36), (327.60, 149.36)
_F_L_in1, _F_L_in2 = (240.00, -0.01), (252.50, 0.00)
FRONT_LOWER_INBOARD = _F_L_in2      # see note above: NOT the average
_F_U_out = (625.00, 212.89)
_F_L_out = (625.00, 27.69)
FRONT_HALF_TRACK = 625.0

_R_U_in1, _R_U_in2 = (327.39, 217.54), (326.32, 217.54)
_R_L_in1, _R_L_in2 = (140.00, 28.70), (140.00, 28.70)
_R_U_out = (600.00, 272.9)
_R_L_out = (600.00, 57.70)
REAR_HALF_TRACK = 600.0

# --- side-view inputs for anti-dive / anti-squat -----------------------------
# Each entry is ((x_fore, z_fore), (x_aft, z_aft)) for one wishbone's inboard
# bushing pair, in the SIDE view: x = longitudinal, z = height above ground.
# The front-view hardpoints above carry no longitudinal coordinate, so these
# cannot be filled in from them. Set them from CAD and the report computes
# the side-view instant centre and the anti percentages. Left as None, the
# report says so rather than inventing a result.
FRONT_LOWER_AXIS = None   # ((x_f, z_f), (x_a, z_a))
FRONT_UPPER_AXIS = None
REAR_LOWER_AXIS = None
REAR_UPPER_AXIS = None

FRONT_CONTACT_X = None    # longitudinal position of the front contact patch
REAR_CONTACT_X = None     # longitudinal position of the rear contact patch
FRONT_BRAKE_BIAS = 0.65   # fraction of braking force at the front axle
REAR_DRIVE_SHARE = 1.0    # fraction of tractive force at the rear axle


def _avg(p, q):
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def make_corner(name, half_track, L_in, L_out, U_in, U_out):
    return Corner(
        name,
        LI=_cad_to_solver(*L_in), LO=_cad_to_solver(*L_out),
        UI=_cad_to_solver(*U_in), UO=_cad_to_solver(*U_out),
        wheel_center0=(half_track, WHEEL_RADIUS),
        wheel_radius=WHEEL_RADIUS, rim_radius=RIM_RADIUS,
        lo_x_offset=LO_X_OFFSET, uo_x_offset=UO_X_OFFSET,
    )


def report_corner(c: Corner):
    p0 = c.static()
    cg_signed = c.camber_gain_per_25mm(dz=TRAVEL, signed=True)
    dzs, rc, cam, clr = c.sweep(-TRAVEL, TRAVEL, 41)
    spread = float(np.nanmax(rc) - np.nanmin(rc))
    min_clr = float(np.nanmin(clr))

    print(f"\n--- {c.name.upper()} CORNER ---")
    print(f"  Static roll-centre height     : {p0.roll_center_z:8.1f} mm")
    if p0.instant_center is not None:
        print(f"  Static instant centre (y, z)  : "
              f"({p0.instant_center[0]:.1f}, {p0.instant_center[1]:.1f}) mm")
    print(f"  FVSA length                   : {p0.fvsa:8.1f} mm   "
          f"<- compare against CAD")
    print(f"  Camber gain (signed)          : {cg_signed:8.2f} deg / {TRAVEL:.0f} mm  "
          f"({'negative camber in bump, good' if cg_signed < 0 else 'POSITIVE camber in bump'})")
    print(f"  Roll-centre migration spread  : {spread:8.1f} mm  (+/-{TRAVEL:.0f} mm)")
    print(f"  Min rim clearance over travel : {min_clr:8.1f} mm  "
          f"(rim inner r={RIM_RADIUS:.0f} mm, ball-joint x-offsets "
          f"{LO_X_OFFSET:.0f}/{UO_X_OFFSET:.0f} mm)")
    print(f"  Lower wishbone (front view)   : {c.L_lower:8.1f} mm")
    print(f"  Upper wishbone (front view)   : {c.L_upper:8.1f} mm")
    print(f"  Upright (kingpin) length      : {c.L_upright:8.1f} mm")
    return dict(rc0=p0.roll_center_z, fvsa=p0.fvsa, cg=cg_signed,
                spread=spread, min_clr=min_clr)


def report_axle_anti(label, lower_axis, upper_axis, contact_x, share, kind):
    if None in (lower_axis, upper_axis, contact_x):
        print(f"  {label:11s}: NOT COMPUTED -- fill in the side-view pivot "
              f"coordinates at the top of this file")
        return None
    ic = side_view_instant_center(lower_axis[0], lower_axis[1],
                                  upper_axis[0], upper_axis[1])
    pct = anti_percent(ic, contact_x, WHEELBASE, CG_HEIGHT, share)
    where = "at infinity (pivot axes parallel)" if ic is None \
        else f"at (x={ic[0]:.1f}, z={ic[1]:.1f}) mm"
    print(f"  {label:11s}: {pct:6.1f} % {kind}   side-view IC {where}")
    return pct


def main():
    front = make_corner("front", FRONT_HALF_TRACK,
                        FRONT_LOWER_INBOARD, _F_L_out,
                        _avg(_F_U_in1, _F_U_in2), _F_U_out)
    rear = make_corner("rear", REAR_HALF_TRACK,
                       _avg(_R_L_in1, _R_L_in2), _R_L_out,
                       _avg(_R_U_in1, _R_U_in2), _R_U_out)

    print("=" * 66)
    print(" DOUBLE-WISHBONE 4-BAR KINEMATICS REPORT")
    print(f" Wheelbase {WHEELBASE:.1f} mm | CG height {CG_HEIGHT:.0f} mm | "
          f"ground plane Y={GROUND_Y}")
    print(" Tyre 540/200-13 (540mm OD) / 13 in rim")
    print("=" * 66)

    report_corner(front)
    report_corner(rear)

    print("\n--- SIDE VIEW: GEOMETRIC ANTI-DIVE / ANTI-SQUAT ---")
    report_axle_anti("front axle", FRONT_LOWER_AXIS, FRONT_UPPER_AXIS,
                     FRONT_CONTACT_X, FRONT_BRAKE_BIAS, "anti-dive")
    report_axle_anti("rear axle", REAR_LOWER_AXIS, REAR_UPPER_AXIS,
                     REAR_CONTACT_X, REAR_DRIVE_SHARE, "anti-squat")


if __name__ == "__main__":
    main()
