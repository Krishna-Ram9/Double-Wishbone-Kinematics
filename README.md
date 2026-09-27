# Double-Wishbone-Kinematics
Custom Python 4-bar linkage solver for double-wishbone suspension kinematics — roll center, camber gain, and migration tracking. Topics: python, vehicle-dynamics, suspension, kinematics, numpy— most importantly — a plot or two (roll center migration curve, camber vs. travel) since this is the one repo that's pure code and the visuals will carry it.
# Double-Wishbone Suspension Kinematics Solver

A from-scratch front-view 4-bar kinematics solver for a double-wishbone
suspension, plus a hardpoint optimizer and an interactive tuning tool.
Computes roll centre, camber gain, swing-arm length, rim clearance, and
side-view anti-dive/anti-squat — and the front/rear roll-centre heights
have been **cross-checked against SolidWorks and match exactly**.

Built for an FSAE-style 13" wheel package (MRF ZTD1 200/540-13, 540 mm
tyre OD), but the solver itself is general-purpose: swap in any
double-wishbone hardpoints and it will solve the linkage.

## Why this exists

Roll centre and camber gain are usually read straight off a CAD
assembly at one static ride height. That's fine for a single number,
but it hides how those numbers move through suspension travel — and
CAD doesn't give you an easy way to search hardpoint-space for a
geometry that hits a *target* roll-centre curve. This project does
both: an exact analytical 4-bar solver (no iterative CAD mate-solving)
swept across ±25 mm of travel, and a least-squares optimizer that
searches hardpoints to hit target RC / camber-gain / RC-migration
values subject to realistic packaging constraints.

## What's in here

| File | What it does |
|---|---|
| `fourbar_solver.py` | Core solver: `Corner` class solves the front-view 4-bar linkage analytically at any travel, returns roll centre, instant centre, camber, FVSA, and rim clearance. Also computes side-view anti-dive/anti-squat. |
| `double_wishbone_report.py` | One-shot printed report for the current validated baseline hardpoints. |
| `interactive_suspension.py` | Menu-driven tool — edit any hardpoint or vehicle parameter by number, see the report recompute instantly. Auto-saves state between runs. |
| `calibrate.py` | Least-squares (trust-region) search over hardpoints to hit target roll centre / camber gain / RC migration spread, subject to arm-length and packaging constraints. |
| `plot_curves.py` | Plots roll centre, camber, and rim clearance vs. wheel travel for front and rear. |

## Results — current baseline (13" wheel package)

Ground plane Y = −107.2 (CAD) · Wheelbase 1564.25 mm · CG height 248 mm

| | Front | Rear |
|---|---|---|
| Static roll centre | **16.9 mm** ✅ matches SolidWorks | **26.4 mm** ✅ matches SolidWorks |
| Camber gain (signed) | −1.08 deg / 25 mm | −0.93 deg / 25 mm |
| RC migration spread (±25 mm) | 42.8 mm | 9.4 mm |
| Min rim clearance over travel | 30.0 mm | 55.0 mm |
| FVSA length | 1330.2 mm | 1541.2 mm |
| Arm lengths (lower / upper / upright) | 373.5 / 304.2 / 185.2 mm | 460.9 / 278.7 / 215.2 mm |

Both corners gain negative camber in bump, which is the desired direction.

![Interactive tool output](<img width="1063" height="655" alt="interactive_tool_menu" src="https://github.com/user-attachments/assets/9f70616f-c1f9-4627-aede-83beeaa5cb97" />
)
![Full kinematics report](<img width="1318" height="717" alt="double_wishbone_report" src="https://github.com/user-attachments/assets/f1cd6d2f-aa49-4011-a234-e454f9660d75" />
)

### Travel curves

Roll centre, camber, and rim clearance swept across ±25 mm of wheel
travel. Front RC crosses zero and goes negative in deep bump; rear RC
stays flatter and holds a small positive migration spread — the kind
of behaviour that's easy to miss from a single static CAD measurement.

![Travel curves](<img width="1872" height="645" alt="travel_curves" src="https://github.com/user-attachments/assets/952574c0-d3c9-4e1d-86dd-646f079d3a4c" />
)

## Hardpoint optimizer

`calibrate.py` runs a `scipy.optimize.least_squares` trust-region
search over 8 hardpoint coordinates per corner (both wishbone inboard
pivots and both ball joints), with constraint residuals for arm length,
ball-joint offset from the wheel-centre plane, and upper/lower pivot
ordering — so the solver can't wander into a geometry that isn't
physically buildable. Converges in well under 50 function evaluations
per corner:

```
ACHIEVED vs TARGET (front RC 44.1 / cg -1.22 / spread 32.0; rear RC 81.9 / cg -1.28 / spread 10.3)
  front: RC  44.10 mm | camber  -1.22 deg/25mm | spread  32.01 mm | FVSA  1178.3 mm | min clearance   5.0 mm
  rear : RC  81.90 mm | camber  -1.28 deg/25mm | spread  10.35 mm | FVSA  1127.1 mm | min clearance   5.2 mm
```

![Calibrator output](<img width="1395" height="297" alt="calibrate_output" src="https://github.com/user-attachments/assets/2f19feba-d77f-46f0-b808-38f635742f26" />
)

## Running it

```bash
pip install -r requirements.txt

python3 double_wishbone_report.py   # one-shot report, validated baseline
python3 interactive_suspension.py   # menu-driven, edit hardpoints live
python3 calibrate.py                # hardpoint optimizer
python3 plot_curves.py              # travel curves -> suspension_curves.png
```

On Windows, if `python`/`pip` aren't recognized, use `py` and
`py -m pip` instead — see [python.org](https://www.python.org/downloads/)
for setup.

## Requirements

- Python 3.9+
- numpy
- scipy (for `calibrate.py`)
- matplotlib (for `plot_curves.py`)

## Notes

- `rim_radius` is currently a simple wheel-diameter/2 proxy (13" / 2),
  not a CAD-measured inner-barrel figure — swap in the real value if
  you have it for a more accurate rim clearance check.
- Side-view anti-dive/anti-squat needs longitudinal (side-view)
  hardpoints, which aren't set in the current baseline — fill them in
  via the `[v]` menu in `interactive_suspension.py` or the constants at
  the top of `double_wishbone_report.py`.
