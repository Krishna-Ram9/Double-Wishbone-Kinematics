"""
plot_curves.py
==============
Plots roll-centre height, camber, and rim clearance vs. wheel travel for
the front and rear corners, using the SAME hardpoints as
interactive_suspension.py (loads suspension_state.json next to this file
if it exists -- i.e. whatever you last edited interactively -- otherwise
falls back to the validated 13" wheel-package baseline).

Run:
    python3 plot_curves.py

Saves suspension_curves.png next to this file and also opens it in a
window if your environment supports it.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from interactive_suspension import build_corner, load_state

HERE = os.path.dirname(os.path.abspath(__file__))
TRAVEL = 25.0
N = 81  # points along the sweep -- smooth curve


def corner_curves(name, cs, ground_y, wheel_radius, rim_radius):
    c = build_corner(name, cs, ground_y, wheel_radius, rim_radius)
    dz, rc, cam, clr = c.sweep(-TRAVEL, TRAVEL, N)
    return dz, rc, cam, clr


def main():
    state = load_state()
    fdz, frc, fcam, fclr = corner_curves(
        "front", state["front"], state["ground_y"],
        state["wheel_radius"], state["rim_radius"])
    rdz, rrc, rcam, rclr = corner_curves(
        "rear", state["rear"], state["ground_y"],
        state["wheel_radius"], state["rim_radius"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(fdz, frc, label="front", color="tab:blue")
    ax.plot(rdz, rrc, label="rear", color="tab:orange")
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("wheel travel (mm, + = bump)")
    ax.set_ylabel("roll-centre height (mm)")
    ax.set_title("Roll centre vs travel")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(fdz, fcam, label="front", color="tab:blue")
    ax.plot(rdz, rcam, label="rear", color="tab:orange")
    ax.axvline(0, color="gray", lw=0.7)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xlabel("wheel travel (mm, + = bump)")
    ax.set_ylabel("camber (deg, - = negative/top-in)")
    ax.set_title("Camber vs travel")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(fdz, fclr, label="front", color="tab:blue")
    ax.plot(rdz, rclr, label="rear", color="tab:orange")
    ax.axvline(0, color="gray", lw=0.7)
    ax.axhline(0, color="red", lw=0.7, ls="--", label="rim contact")
    ax.set_xlabel("wheel travel (mm, + = bump)")
    ax.set_ylabel("rim clearance (mm)")
    ax.set_title("Rim clearance vs travel")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Double-wishbone travel curves (13\" wheel package)")
    fig.tight_layout()

    out_path = os.path.join(HERE, "suspension_curves.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
