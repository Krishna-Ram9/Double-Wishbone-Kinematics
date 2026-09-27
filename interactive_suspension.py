"""
interactive_suspension.py
==========================
Interactive front-end for the fourbar_solver double-wishbone solver.

Pick a hardpoint or vehicle parameter by number, type in its new value,
and the front/rear roll centre, IC-to-contact-patch swing-arm length,
camber gain, migration spread, rim clearance and anti-dive/anti-squat
are recomputed and printed immediately -- no editing source, no
restarting between tweaks.

Values auto-save to suspension_state.json next to this file after
every edit, so a session picks up right where the last one left off.
Use the "d" menu option to go back to the 13" wheel-package baseline
hardpoints below.

Run:
    python3 interactive_suspension.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from fourbar_solver import (Corner, LinkageError, anti_percent,
                             side_view_instant_center)

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(HERE, "suspension_state.json")
TRAVEL = 25.0

# Baseline -- 13" wheel package (MRF ZTD1 200/540-13: 540mm tyre OD, 13"
# rim). Outboard (upright) points are the CAD-measured values; inboard
# points carry over unchanged from the earlier 18"-wheel SolidWorks model
# ("the inner points of the chassis are the same"). rim_radius is the
# simple 13" wheel-diameter/2 proxy used throughout this toolchain, not a
# CAD-measured inner-barrel figure -- replace it if you have the real one.
# (CAD frame: X = lateral, Y = vertical; inboard bushing pairs pre-resolved
# to the single point double_wishbone_report.py uses for each axis.)
# VALIDATED against SolidWorks: front RC 16.9 mm, rear RC 26.4 mm.
DEFAULT_STATE = {
    "ground_y": -107.2,
    # wheelbase = rear Z - front Z = -640.75 -> -2205 = 1564.25 mm (CAD upright points)
    "wheelbase": 1564.25,
    "cg_height": 248.0,
    "wheel_radius": 540.0 / 2,
    "rim_radius": (13 * 25.4) / 2,
    "front": {
        "half_track": 625.0,
        "lo_x_offset": 0.0, "uo_x_offset": 0.0,
        "LI": [252.50, 0.00], "LO": [625.00, 27.69],
        "UI": [327.53, 149.36], "UO": [625.00, 212.89],
    },
    "rear": {
        "half_track": 600.0,
        "lo_x_offset": 0.0, "uo_x_offset": 0.0,
        "LI": [140.00, 28.70], "LO": [600.00, 57.70],
        "UI": [326.855, 217.54], "UO": [600.00, 272.9],
    },
    "side_view": {
        "front_lower": None, "front_upper": None,
        "rear_lower": None, "rear_upper": None,
        "front_contact_x": None, "rear_contact_x": None,
        "front_brake_bias": 0.65, "rear_drive_share": 1.0,
    },
}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return json.loads(json.dumps(DEFAULT_STATE))  # deep copy


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def cad_to_solver(pt, ground_y):
    """CAD (X lateral, Y vertical) -> solver (y lateral, z above ground)."""
    return (pt[0], pt[1] - ground_y)


def build_corner(name, cs, ground_y, wheel_radius, rim_radius):
    return Corner(
        name,
        LI=cad_to_solver(cs["LI"], ground_y),
        LO=cad_to_solver(cs["LO"], ground_y),
        UI=cad_to_solver(cs["UI"], ground_y),
        UO=cad_to_solver(cs["UO"], ground_y),
        wheel_center0=(cs["half_track"], wheel_radius),
        wheel_radius=wheel_radius, rim_radius=rim_radius,
        lo_x_offset=cs["lo_x_offset"], uo_x_offset=cs["uo_x_offset"],
    )


def report_corner(name, cs, ground_y, wheel_radius, rim_radius):
    print(f"\n--- {name.upper()} ---")
    try:
        c = build_corner(name, cs, ground_y, wheel_radius, rim_radius)
    except LinkageError as e:
        print(f"  cannot assemble at design position: {e}")
        return
    try:
        p0 = c.static()
        cg = c.camber_gain_per_25mm(dz=TRAVEL, signed=True)
        dzs, rc, cam, clr = c.sweep(-TRAVEL, TRAVEL, 41)
        spread = float(np.nanmax(rc) - np.nanmin(rc))
        min_clr = float(np.nanmin(clr))
    except LinkageError as e:
        print(f"  does not solve over +/-{TRAVEL:.0f} mm travel: {e}")
        return

    if p0.instant_center is not None:
        icy, icz = p0.instant_center
        wy, wz = p0.wheel_center
        # contact patch taken directly below the wheel centre, at ground level
        ic_to_contact = math.hypot(icy - wy, icz)
        print(f"  IC (to contact patch, swing-arm)  : {ic_to_contact:8.2f} mm")
    else:
        print(f"  IC (to contact patch, swing-arm)  :      n/a (IC at infinity)")
    print(f"  Roll centre                       : {p0.roll_center_z:8.2f} mm above ground")
    print(f"  FVSA (lateral, IC to wheel-centre) : {p0.fvsa:8.2f} mm  <- compare to CAD")
    print(f"  Camber gain                       : {cg:+7.3f} deg / {TRAVEL:.0f}mm "
          f"({'negative in bump, good' if cg < 0 else 'POSITIVE in bump'})")
    print(f"  RC migration spread                : {spread:8.2f} mm  (+/-{TRAVEL:.0f} mm)")
    print(f"  Min rim clearance                  : {min_clr:8.2f} mm")
    print(f"  Arm lengths                        : lower {c.L_lower:.1f}  upper {c.L_upper:.1f}  "
          f"upright {c.L_upright:.1f} mm")


def report_side_view(state):
    sv = state["side_view"]
    print("\n--- SIDE VIEW: ANTI-DIVE / ANTI-SQUAT ---")
    for label, lower_key, upper_key, contact_key, share_key, kind in [
        ("front axle", "front_lower", "front_upper", "front_contact_x",
         "front_brake_bias", "anti-dive"),
        ("rear axle", "rear_lower", "rear_upper", "rear_contact_x",
         "rear_drive_share", "anti-squat"),
    ]:
        lower, upper, contact = sv[lower_key], sv[upper_key], sv[contact_key]
        if lower is None or upper is None or contact is None:
            print(f"  {label:11s}: not set -- fill in side-view axis points and contact-patch X ([v] menu)")
            continue
        ic = side_view_instant_center(tuple(lower[0]), tuple(lower[1]),
                                       tuple(upper[0]), tuple(upper[1]))
        pct = anti_percent(ic, contact, state["wheelbase"], state["cg_height"], sv[share_key])
        where = "at infinity" if ic is None else f"(x={ic[0]:.1f}, z={ic[1]:.1f}) mm"
        print(f"  {label:11s}: {pct:6.1f} % {kind}   side-view IC {where}")


def full_report(state):
    report_corner("front", state["front"], state["ground_y"],
                  state["wheel_radius"], state["rim_radius"])
    report_corner("rear", state["rear"], state["ground_y"],
                  state["wheel_radius"], state["rim_radius"])
    report_side_view(state)


# -- parameter registry ----------------------------------------------------

def hardpoint_params(state):
    params = []
    for corner in ("front", "rear"):
        for point in ("LI", "LO", "UI", "UO"):
            for axis, idx in (("X", 0), ("Y", 1)):
                params.append((f"{corner} {point} {axis} (CAD)", corner, point, axis, idx))
    return params


def scalar_params():
    return [
        ("Ground plane Y (CAD)", None, "ground_y"),
        ("Wheelbase (mm)", None, "wheelbase"),
        ("CG height (mm)", None, "cg_height"),
        ("Wheel (tyre) radius (mm)", None, "wheel_radius"),
        ("Rim inner-barrel radius (mm)", None, "rim_radius"),
        ("Front half-track (mm)", ("front", "half_track"), None),
        ("Rear half-track (mm)", ("rear", "half_track"), None),
        ("Front lower ball-joint fore/aft offset (mm)", ("front", "lo_x_offset"), None),
        ("Front upper ball-joint fore/aft offset (mm)", ("front", "uo_x_offset"), None),
        ("Rear lower ball-joint fore/aft offset (mm)", ("rear", "lo_x_offset"), None),
        ("Rear upper ball-joint fore/aft offset (mm)", ("rear", "uo_x_offset"), None),
    ]


def _get_scalar(state, path, key):
    return state[path[0]][path[1]] if path else state[key]


def _set_scalar(state, path, key, value):
    if path:
        state[path[0]][path[1]] = value
    else:
        state[key] = value


def print_hardpoint_menu(state):
    hp = hardpoint_params(state)
    print("\n  HARDPOINTS (CAD frame: X = lateral, Y = vertical)")
    for i, (label, corner, point, axis, idx) in enumerate(hp, 1):
        val = state[corner][point][idx]
        print(f"   {i:2d}) {label:28s} = {val:9.3f}")
    sp = scalar_params()
    offset = len(hp)
    print("\n  VEHICLE / CORNER PARAMETERS")
    for j, (label, path, key) in enumerate(sp, offset + 1):
        print(f"   {j:2d}) {label:44s} = {_get_scalar(state, path, key):9.3f}")
    return hp, sp


def edit_hardpoint(state):
    hp, sp = print_hardpoint_menu(state)
    total = len(hp) + len(sp)
    choice = input(f"\n  Pick a parameter to edit [1-{total}], or Enter to cancel: ").strip()
    if not choice:
        return
    try:
        n = int(choice)
        assert 1 <= n <= total
    except (ValueError, AssertionError):
        print("  not a valid choice")
        return

    if n <= len(hp):
        label, corner, point, axis, idx = hp[n - 1]
        cur = state[corner][point][idx]
        raw = input(f"  new value for {label} (current {cur:.3f}), Enter to cancel: ").strip()
        if not raw:
            return
        try:
            state[corner][point][idx] = float(raw)
        except ValueError:
            print("  not a number")
            return
    else:
        label, path, key = sp[n - 1 - len(hp)]
        cur = _get_scalar(state, path, key)
        raw = input(f"  new value for {label} (current {cur:.3f}), Enter to cancel: ").strip()
        if not raw:
            return
        try:
            v = float(raw)
        except ValueError:
            print("  not a number")
            return
        _set_scalar(state, path, key, v)

    save_state(state)
    full_report(state)


def edit_side_view(state):
    sv = state["side_view"]
    fields = [
        ("front_lower", "Front lower wishbone inboard axis  [[x_fore,z_fore],[x_aft,z_aft]]"),
        ("front_upper", "Front upper wishbone inboard axis"),
        ("rear_lower", "Rear lower wishbone inboard axis"),
        ("rear_upper", "Rear upper wishbone inboard axis"),
        ("front_contact_x", "Front contact-patch X"),
        ("rear_contact_x", "Rear contact-patch X"),
        ("front_brake_bias", "Front brake bias (0-1)"),
        ("rear_drive_share", "Rear drive share (0-1)"),
    ]
    print("\n  SIDE VIEW (longitudinal x, height z -- needed for anti-dive/anti-squat)")
    for i, (key, label) in enumerate(fields, 1):
        print(f"   {i}) {label} = {sv[key]}")
    choice = input(f"\n  Pick a field to edit [1-{len(fields)}], or Enter to cancel: ").strip()
    if not choice:
        return
    try:
        n = int(choice)
        assert 1 <= n <= len(fields)
    except (ValueError, AssertionError):
        print("  not a valid choice")
        return

    key, label = fields[n - 1]
    if key in ("front_contact_x", "rear_contact_x", "front_brake_bias", "rear_drive_share"):
        raw = input(f"  new value for {label} (current {sv[key]}), Enter to cancel: ").strip()
        if not raw:
            return
        try:
            sv[key] = float(raw)
        except ValueError:
            print("  not a number")
            return
    else:
        raw = input(f"  enter as x_fore,z_fore,x_aft,z_aft (current {sv[key]}), "
                    f"or blank to clear/cancel: ").strip()
        if not raw:
            sv[key] = None
        else:
            try:
                xf, zf, xa, za = (float(v) for v in raw.split(","))
                sv[key] = [[xf, zf], [xa, za]]
            except ValueError:
                print("  need four comma-separated numbers")
                return

    save_state(state)
    report_side_view(state)


def main():
    state = load_state()
    print("Interactive double-wishbone report -- state auto-saves to")
    print(f"  {STATE_FILE}")
    full_report(state)
    while True:
        print("\n" + "=" * 60)
        print("  [e] edit a hardpoint/vehicle parameter")
        print("  [v] edit side-view (anti-dive/squat) inputs")
        print("  [r] reprint full report")
        print("  [d] reset everything to defaults")
        print("  [q] quit")
        cmd = input("> ").strip().lower()
        if cmd == "e":
            edit_hardpoint(state)
        elif cmd == "v":
            edit_side_view(state)
        elif cmd == "r":
            full_report(state)
        elif cmd == "d":
            confirm = input("  reset ALL values to the validated defaults? [y/N]: ").strip().lower()
            if confirm == "y":
                state = json.loads(json.dumps(DEFAULT_STATE))
                save_state(state)
                full_report(state)
        elif cmd == "q":
            print("state saved -- run again any time to pick up where you left off")
            break
        else:
            print("  not a recognised command")


if __name__ == "__main__":
    main()
