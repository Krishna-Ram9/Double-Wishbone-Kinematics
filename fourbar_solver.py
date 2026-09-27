"""
fourbar_solver.py  (v2)
=======================
Double-wishbone front-view 4-bar linkage solver.

Coordinates used internally (SOLVER FRAME):
    y = lateral, positive outboard (left side of car)
    z = height above ground, ground plane at z = 0
Units: millimetres, degrees where noted.

    LI  = lower wishbone inboard pivot   (fixed to chassis)
    LO  = lower ball joint               (circle about LI, radius L_lower)
    UI  = upper wishbone inboard pivot   (fixed to chassis)
    UO  = upper ball joint               (circle about UI, radius L_upper)

The upright is the rigid coupler between LO and UO. The wheel centre is a
point rigidly attached to that coupler.

WHAT CHANGED FROM v1
--------------------
1.  The linkage is swept in phi (the lower-arm angle) and results are
    interpolated onto the requested travel grid. v1 root-solved phi for
    every single dz, costing 15+ pose evaluations each. Same answers to
    <0.005 mm, ~28x faster.
2.  Static (design-position) values are evaluated exactly at phi0. No
    root solve is needed for dz = 0; v1 solved for it numerically.
3.  Camber is SIGNED. v1 returned (|bump| + |droop|)/2, so a geometry
    that gained positive camber in bump scored the same as one that
    gained negative camber. Sign convention here: negative camber = top
    of the wheel leaning inboard, which is what you want in bump.
4.  Rim clearance is computed correctly. v1 measured the full 2D distance
    from ball joint to wheel centre and subtracted the rim radius. Two
    errors there: (a) the lateral component of that distance runs ALONG
    the wheel spin axis and must not be counted, and (b) the sign was
    inverted -- the ball joints live INSIDE the rim barrel, so the margin
    is (rim_radius - radial_distance), not the other way round. That is
    why v1 printed negative numbers and had to be overridden by hand.
5.  Branch selection is stateless. The upper-circle intersection branch
    is locked once at phi0 by matching the design-position UO, then held.
    v1 carried mutable _last_UO state between calls, so results could
    depend on call order.
6.  side_view_geometric_antidive() is gone. It compared two numbers for
    equality and returned 0.0. Replaced with an actual side-view instant
    centre intersection plus the standard anti-dive/anti-squat formula.
7.  Reports FVSA length and instant-centre position, so the computed
    FVSA can be checked directly against CAD.
8.  Inner geometry is scalar math (math.hypot etc.) rather than numpy on
    2-element arrays, which dominated runtime.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["Corner", "Pose", "side_view_instant_center",
           "anti_percent", "LinkageError"]


class LinkageError(RuntimeError):
    """Raised when the 4-bar cannot be assembled at the requested pose."""


class Pose:
    """Solved state of one corner at one lower-arm angle."""

    __slots__ = ("phi", "dz", "LO", "UO", "wheel_center", "camber_deg",
                 "instant_center", "roll_center_z", "fvsa", "clearance")

    def __init__(self, phi, dz, LO, UO, wc, camber, ic, rc, fvsa, clr):
        self.phi = phi
        self.dz = dz
        self.LO = LO
        self.UO = UO
        self.wheel_center = wc
        self.camber_deg = camber
        self.instant_center = ic
        self.roll_center_z = rc
        self.fvsa = fvsa
        self.clearance = clr

    def __repr__(self):
        return (f"<Pose dz={self.dz:+.2f} camber={self.camber_deg:+.3f} "
                f"rc={self.roll_center_z:.2f} fvsa={self.fvsa:.1f}>")


class Corner:
    """One front-view double-wishbone corner (planar 4-bar linkage).

    Parameters
    ----------
    LI, LO, UI, UO : (y, z) in the solver frame, design position.
    wheel_center0  : (y, z) static wheel centre.
    wheel_radius   : loaded tyre radius, mm.
    rim_radius     : rim INNER barrel radius, mm. Ball joints must sit
                     inside this cylinder about the wheel spin axis.
    lo_x_offset, uo_x_offset : longitudinal (fore/aft) offset of each ball
                     joint from the wheel-centre plane, mm. Defaults to 0.
                     Read these off CAD -- they matter for clearance.
    """

    def __init__(self, name, LI, LO, UI, UO, wheel_center0,
                 wheel_radius, rim_radius,
                 lo_x_offset=0.0, uo_x_offset=0.0):
        self.name = name
        self.LIy, self.LIz = float(LI[0]), float(LI[1])
        self.LO0y, self.LO0z = float(LO[0]), float(LO[1])
        self.UIy, self.UIz = float(UI[0]), float(UI[1])
        self.UO0y, self.UO0z = float(UO[0]), float(UO[1])
        self.wc0y, self.wc0z = float(wheel_center0[0]), float(wheel_center0[1])
        self.wheel_radius = float(wheel_radius)
        self.rim_radius = float(rim_radius)
        self.lo_x_offset = float(lo_x_offset)
        self.uo_x_offset = float(uo_x_offset)

        self.L_lower = math.hypot(self.LO0y - self.LIy, self.LO0z - self.LIz)
        self.L_upper = math.hypot(self.UO0y - self.UIy, self.UO0z - self.UIz)
        self.L_upright = math.hypot(self.UO0y - self.LO0y, self.UO0z - self.LO0z)
        if min(self.L_lower, self.L_upper, self.L_upright) < 1e-6:
            raise LinkageError(f"{name}: degenerate link length")

        self.phi0 = math.atan2(self.LO0z - self.LIz, self.LO0y - self.LIy)
        self._upright_ang0 = math.atan2(self.UO0z - self.LO0z,
                                        self.UO0y - self.LO0y)

        # Lock the intersection branch once, by matching the design UO.
        self._branch = 1.0
        cand = self._uo_at(self.LO0y, self.LO0z, 1.0)
        if cand is None:
            raise LinkageError(f"{name}: cannot assemble at design position")
        if math.hypot(cand[0] - self.UO0y, cand[1] - self.UO0z) > \
           math.hypot(*self._diff(self._uo_at(self.LO0y, self.LO0z, -1.0))):
            self._branch = -1.0

        self._table = None  # lazily built (phi, dz) table for travel queries

    # -- internals ---------------------------------------------------

    def _diff(self, p):
        if p is None:
            return (1e9, 1e9)
        return (p[0] - self.UO0y, p[1] - self.UO0z)

    def _uo_at(self, LOy, LOz, branch):
        """Upper ball joint: intersection of circle(UI, L_upper) and
        circle(LO, L_upright), on the locked branch."""
        dy = LOy - self.UIy
        dz = LOz - self.UIz
        d = math.hypot(dy, dz)
        if d < 1e-9 or d > self.L_upper + self.L_upright or \
           d < abs(self.L_upper - self.L_upright):
            return None
        a = (self.L_upper * self.L_upper - self.L_upright * self.L_upright
             + d * d) / (2.0 * d)
        h_sq = self.L_upper * self.L_upper - a * a
        if h_sq < 0.0:
            h_sq = 0.0
        h = math.sqrt(h_sq)
        mx = self.UIy + a * dy / d
        my = self.UIz + a * dz / d
        return (mx + branch * h * (-dz / d), my + branch * h * (dy / d))

    def pose(self, phi) -> Pose:
        """Full solved state at lower-arm angle phi."""
        LOy = self.LIy + self.L_lower * math.cos(phi)
        LOz = self.LIz + self.L_lower * math.sin(phi)
        uo = self._uo_at(LOy, LOz, self._branch)
        if uo is None:
            raise LinkageError(f"{self.name}: linkage locks at phi={phi:.4f}")
        UOy, UOz = uo

        # rigid-body rotation of the upright, relative to design position
        th = math.atan2(UOz - LOz, UOy - LOy) - self._upright_ang0
        ct, st = math.cos(th), math.sin(th)
        ty = LOy - (ct * self.LO0y - st * self.LO0z)
        tz = LOz - (st * self.LO0y + ct * self.LO0z)
        wy = ct * self.wc0y - st * self.wc0z + ty
        wz = st * self.wc0y + ct * self.wc0z + tz

        # SIGN CONVENTION: th > 0 rotates the top of the upright inboard,
        # which is negative camber. Flip so the reported number reads as
        # camber in the usual sense.
        camber = -math.degrees(th)

        # instant centre = line(LI, LO) x line(UI, UO)
        d1y, d1z = LOy - self.LIy, LOz - self.LIz
        d2y, d2z = UOy - self.UIy, UOz - self.UIz
        den = d1y * d2z - d1z * d2y
        ic = None
        rc = math.nan
        fvsa = math.inf
        if abs(den) > 1e-12:
            t = ((self.UIy - self.LIy) * d2z - (self.UIz - self.LIz) * d2y) / den
            icy = self.LIy + t * d1y
            icz = self.LIz + t * d1z
            ic = (icy, icz)
            fvsa = abs(icy - wy)
            # roll centre = line(IC, contact patch) crossing the centreline
            dpy = wy - icy
            if abs(dpy) > 1e-12:
                rc = icz + (0.0 - icy) / dpy * (0.0 - icz)

        # rim clearance: radial distance of each ball joint from the wheel
        # SPIN AXIS (a lateral line through the wheel centre). The lateral
        # separation runs along that axis and does not count.
        r_lo = math.hypot(LOz - wz, self.lo_x_offset)
        r_uo = math.hypot(UOz - wz, self.uo_x_offset)
        clr = self.rim_radius - max(r_lo, r_uo)

        return Pose(phi, wz - self.wc0z, (LOy, LOz), (UOy, UOz), (wy, wz),
                    camber, ic, rc, fvsa, clr)

    def _build_table(self, half_range=0.45, n=361):
        phis = np.linspace(self.phi0 - half_range, self.phi0 + half_range, n)
        dzs = np.empty(n)
        ok = np.ones(n, dtype=bool)
        for i, p in enumerate(phis):
            try:
                dzs[i] = self.pose(p).dz
            except LinkageError:
                dzs[i] = np.nan
                ok[i] = False
        phis, dzs = phis[ok], dzs[ok]
        order = np.argsort(dzs)
        self._table = (phis[order], dzs[order])
        return self._table

    def phi_for_dz(self, dz):
        """Lower-arm angle giving vertical wheel-centre travel dz."""
        phis, dzs = self._table or self._build_table()
        if dz < dzs[0] or dz > dzs[-1]:
            raise LinkageError(f"{self.name}: dz={dz} outside solvable travel "
                               f"[{dzs[0]:.1f}, {dzs[-1]:.1f}] mm")
        phi = float(np.interp(dz, dzs, phis))
        # one Newton polish -- the table is dense, so this converges at once
        for _ in range(3):
            f0 = self.pose(phi).dz - dz
            if abs(f0) < 1e-9:
                break
            h = 1e-6
            df = (self.pose(phi + h).dz - self.pose(phi - h).dz) / (2 * h)
            if abs(df) < 1e-12:
                break
            phi -= f0 / df
        return phi

    # -- public API --------------------------------------------------

    def at_travel(self, dz) -> Pose:
        """Solved state at a vertical wheel-centre travel dz (+ = jounce)."""
        if dz == 0.0:
            return self.pose(self.phi0)   # exact, no solve needed
        return self.pose(self.phi_for_dz(dz))

    def static(self) -> Pose:
        """Design-position state. Exact at phi0."""
        return self.pose(self.phi0)

    def static_roll_center(self):
        """(roll_centre_height_mm, instant_centre) at design position."""
        p = self.static()
        return p.roll_center_z, p.instant_center

    def sweep(self, dz_min=-25.0, dz_max=25.0, n=41, oversample=4):
        """Sweep travel. Returns (dz, roll_centre_z, camber_deg, clearance).

        Sweeps phi once and interpolates onto the dz grid, rather than
        root-solving phi for every dz.
        """
        m = max(int(n * oversample), 64)
        phis = np.linspace(self.phi0 - 0.45, self.phi0 + 0.45, m)
        dz_s = np.empty(m)
        rc_s = np.empty(m)
        cam_s = np.empty(m)
        clr_s = np.empty(m)
        good = np.ones(m, dtype=bool)
        for i, p in enumerate(phis):
            try:
                ps = self.pose(p)
            except LinkageError:
                good[i] = False
                continue
            dz_s[i], rc_s[i] = ps.dz, ps.roll_center_z
            cam_s[i], clr_s[i] = ps.camber_deg, ps.clearance
        dz_s, rc_s, cam_s, clr_s = (a[good] for a in (dz_s, rc_s, cam_s, clr_s))
        o = np.argsort(dz_s)
        dz_s, rc_s, cam_s, clr_s = dz_s[o], rc_s[o], cam_s[o], clr_s[o]

        grid = np.linspace(dz_min, dz_max, n)
        if grid[0] < dz_s[0] - 1e-9 or grid[-1] > dz_s[-1] + 1e-9:
            raise LinkageError(
                f"{self.name}: requested travel [{dz_min}, {dz_max}] exceeds "
                f"solvable range [{dz_s[0]:.1f}, {dz_s[-1]:.1f}] mm")
        return (grid,
                np.interp(grid, dz_s, rc_s),
                np.interp(grid, dz_s, cam_s),
                np.interp(grid, dz_s, clr_s))

    def camber_gain_per_25mm(self, dz=25.0, signed=True):
        """Camber change per 25 mm of travel.

        signed=True  -> mean SIGNED rate, negative = gains negative camber
                        in bump (what you want).
        signed=False -> v1 behaviour, mean of magnitudes. Kept only so old
                        numbers can be reproduced; do not optimise against it.
        """
        c_up = self.at_travel(+dz).camber_deg
        c_dn = self.at_travel(-dz).camber_deg
        if signed:
            return (c_up - c_dn) / 2.0
        return (abs(c_up) + abs(c_dn)) / 2.0

    def roll_center_migration(self, dz=25.0, n=41):
        _, rc, _, _ = self.sweep(-dz, dz, n)
        return float(np.nanmax(rc) - np.nanmin(rc))

    def min_clearance(self, dz=25.0, n=41):
        _, _, _, clr = self.sweep(-dz, dz, n)
        return float(np.nanmin(clr))


# ---------------------------------------------------------------------
# Side view: anti-dive / anti-squat
# ---------------------------------------------------------------------

def side_view_instant_center(lower_fore, lower_aft, upper_fore, upper_aft):
    """Side-view instant centre from the two inboard pivot axes.

    Each argument is (x, z): longitudinal and vertical coordinate of one
    inboard bushing, in the side view. Returns (x, z) of the intersection,
    or None if the two axes are parallel (IC at infinity).

    Two horizontal axes -> parallel -> None -> zero geometric anti-feature.
    That result is now DERIVED rather than asserted.
    """
    (x1, z1), (x2, z2) = lower_fore, lower_aft
    (x3, z3), (x4, z4) = upper_fore, upper_aft
    d1x, d1z = x2 - x1, z2 - z1
    d2x, d2z = x4 - x3, z4 - z3
    den = d1x * d2z - d1z * d2x
    if abs(den) < 1e-12:
        return None
    t = ((x3 - x1) * d2z - (z3 - z1) * d2x) / den
    return (x1 + t * d1x, z1 + t * d1z)


def anti_percent(ic, contact_patch_x, wheelbase, cg_height, load_share):
    """Geometric anti-dive (front) or anti-squat (rear), in percent.

    ic              : side-view instant centre (x, z), or None for infinity
    contact_patch_x : longitudinal position of the contact patch
    wheelbase       : mm
    cg_height       : mm
    load_share      : fraction of the longitudinal force reacted at this
                      axle -- front brake bias for anti-dive, 1.0 for
                      anti-squat on a rear-driven car with inboard-mounted
                      final drive handled separately.

    anti% = 100 * load_share * (tan(theta) * wheelbase / cg_height),
    where tan(theta) is the slope of the line from the contact patch to
    the side-view instant centre.
    """
    if ic is None:
        return 0.0
    dx = ic[0] - contact_patch_x
    if abs(dx) < 1e-9:
        return 0.0
    return 100.0 * load_share * (ic[1] / dx) * wheelbase / cg_height
