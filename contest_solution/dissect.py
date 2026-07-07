"""Exact-area dissection engine v2 (CAMPAIGN_GOLDEN, G2-G4).

Golden layouts are near-perfect tessellations (docs/CAMPAIGN_GOLDEN.md §2), so
this engine *dissects* the die instead of packing rigid rectangles. Soft-block
dims are derived from the structure (w = area/height), making areas exact and
utilization ≈ 1 wherever the structure is unobstructed.

Layout structure (mirrors the golden frame-plus-interior organization):

        ┌───────────────── top band (top-required) ─────────────────┐
        │ left │        interior: exact rows, obstacle-aware        │ right │
        │ col  │  (slabs cut at preplaced y-edges; free segments    │ col   │
        │(left-│   filled best-fit; unobstructed span = flexible    │(right-│
        │ req) │   exact rows)                                      │ req)  │
        └──────────────── bottom band (bottom-required) ────────────┘

- bands span the full die width -> their members touch y_min / y_max;
  corner codes sit at the band ends (touching both edges).
- columns span the mid region -> members touch x_min / x_max.
- clusters are contiguous runs (single row segment or two stacked lanes) ->
  members tile a connected region -> abutment by construction.
- MIB groups (equal target areas) become adjacent identical slots.
- fixed-shape blocks keep exact dims; their row absorbs the slack.
- preplaced blocks are pinned obstacles; interior rows are cut around them.

Everything is stdlib Python, deterministic, list-based (no torch).
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

Rect = Tuple[float, float, float, float]
_EPS = 1e-9


# ---------------------------------------------------------------------------
# case parsing
# ---------------------------------------------------------------------------

class Case:
    def __init__(self, n, areas, constraints, target_positions):
        self.n = n
        self.area = [float(areas[i]) for i in range(n)]
        self.fixed = [False] * n
        self.preplaced = [False] * n
        self.mib = [0] * n
        self.cluster = [0] * n
        self.boundary = [0] * n
        if constraints is not None:
            for i in range(n):
                row = constraints[i]
                nc = len(row)
                self.fixed[i] = nc > 0 and float(row[0]) != 0
                self.preplaced[i] = nc > 1 and float(row[1]) != 0
                self.mib[i] = int(float(row[2])) if nc > 2 else 0
                self.cluster[i] = int(float(row[3])) if nc > 3 else 0
                self.boundary[i] = int(float(row[4])) if nc > 4 else 0
        self.tp = ([[float(v) for v in target_positions[i]] for i in range(n)]
                   if target_positions is not None else None)

    def fixed_dims(self, i):
        if self.tp is None:
            return None
        w, h = self.tp[i][2], self.tp[i][3]
        return (w, h) if (w != -1 and h != -1) else None

    def pre_rect(self, i):
        if self.tp is None:
            return None
        x, y, w, h = self.tp[i]
        return (x, y, w, h) if -1 not in (x, y, w, h) else None

    def block_area(self, i):
        fd = self.fixed_dims(i)
        return fd[0] * fd[1] if fd else self.area[i]


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

class Unit:
    __slots__ = ("blocks", "kind", "area", "boundary", "fixed_h", "fixed_w")

    def __init__(self, blocks, kind, case):
        self.blocks = blocks
        self.kind = kind          # 'block' | 'mib' | 'cluster'
        self.area = sum(case.block_area(b) for b in blocks)
        self.boundary = 0
        for b in blocks:
            self.boundary |= case.boundary[b]
        dims = [case.fixed_dims(b) for b in blocks if case.fixed_dims(b)]
        self.fixed_h = max((d[1] for d in dims), default=0.0)
        self.fixed_w = sum(d[0] for d in dims)


def build_units(case, movable):
    used, units = set(), []
    cl = {}
    for i in movable:
        if case.cluster[i] > 0:
            cl.setdefault(case.cluster[i], []).append(i)
    for gid in sorted(cl):
        if len(cl[gid]) > 1:
            units.append(Unit(cl[gid], 'cluster', case))
            used.update(cl[gid])
    mg = {}
    for i in movable:
        if i not in used and case.mib[i] > 0 and case.fixed_dims(i) is None:
            mg.setdefault(case.mib[i], []).append(i)
    for gid in sorted(mg):
        blocks = mg[gid]
        if len(blocks) > 1:
            amin = min(case.area[b] for b in blocks)
            amax = max(case.area[b] for b in blocks)
            if amax / amin <= 1.005:
                units.append(Unit(blocks, 'mib', case))
                used.update(blocks)
    for i in movable:
        if i not in used:
            units.append(Unit([i], 'block', case))
    return units


# ---------------------------------------------------------------------------
# geometric helpers
# ---------------------------------------------------------------------------

def _free_intervals(lo, hi, blocked):
    """Subtract blocked [a,b) intervals from [lo,hi); return free intervals."""
    ivs = [(lo, hi)]
    for a, b in blocked:
        nxt = []
        for s, e in ivs:
            if b <= s + _EPS or a >= e - _EPS:
                nxt.append((s, e))
            else:
                if a > s + _EPS:
                    nxt.append((s, a))
                if b < e - _EPS:
                    nxt.append((b, e))
        ivs = nxt
    return [iv for iv in ivs if iv[1] - iv[0] > 1e-6]


def _edge_order(case, blocks):
    return sorted(blocks, key=lambda b: (0 if case.boundary[b] & 1 else
                                         (2 if case.boundary[b] & 2 else 1)))


# ---------------------------------------------------------------------------
# realization primitives (all exact-fill)
# ---------------------------------------------------------------------------

def _place_unit_in_row(case, u, x, y, h, out):
    """Place unit u at x in row band [y, y+h). Returns the new x cursor."""
    if u.kind == 'mib':
        a = min(case.area[b] for b in u.blocks)
        w = a / h
        for b in u.blocks:
            out[b] = (x, y, w, h)
            x += w
        return x
    if u.kind == 'cluster':
        return _place_cluster(case, u, x, y, h, out)
    b = u.blocks[0]
    fd = case.fixed_dims(b)
    if fd:
        out[b] = (x, y, fd[0], fd[1])
        return x + fd[0]
    w = case.area[b] / h
    out[b] = (x, y, w, h)
    return x + w


def _place_cluster(case, u, x, y, h, out):
    blocks = list(u.blocks)
    fixed = [b for b in blocks if case.fixed_dims(b)]
    soft = [b for b in blocks if not case.fixed_dims(b)]
    if fixed or len(blocks) <= 2:
        cx = x
        for b in sorted(fixed, key=lambda b: -case.fixed_dims(b)[1]):
            w, hh = case.fixed_dims(b)
            out[b] = (cx, y, w, hh)
            cx += w
        for b in _edge_order(case, soft):
            w = case.area[b] / h
            out[b] = (cx, y, w, h)
            cx += w
        return cx
    # two stacked lanes of equal width => connected tiling, milder aspects
    blocks.sort(key=lambda b: -case.area[b])
    lane, la = [[], []], [0.0, 0.0]
    for b in blocks:
        k = 0 if la[0] <= la[1] else 1
        lane[k].append(b)
        la[k] += case.area[b]
    if not lane[1]:
        lane[1] = [lane[0].pop()]
        la = [sum(case.area[b] for b in lane[0]),
              sum(case.area[b] for b in lane[1])]
    wc = u.area / h
    h0 = h * la[0] / u.area
    for lane_blocks, ly, lh in ((lane[0], y, h0), (lane[1], y + h0, h - h0)):
        cx = x
        for b in lane_blocks:
            out[b] = (cx, ly, case.area[b] / lh, lh)
            cx += case.area[b] / lh
    return x + wc


def _fill_rows(case, units, x0, W, y0, out, min_h=0.0):
    """Stack exact-fill rows upward from y0 spanning [x0, x0+W). Row heights
    derive from membership; every row spans the width exactly (soft blocks
    absorb). Returns y after the last row."""
    y = y0
    i = 0
    units = list(units)
    while i < len(units):
        # accumulate a row: aim for the near-square height of its members,
        # never below the tallest fixed member
        row = []
        area = 0.0
        fixh = 0.0
        fixw = 0.0
        while i < len(units):
            u = units[i]
            trial_area = area + u.area
            trial_fixh = max(fixh, u.fixed_h)
            trial_fixw = fixw + u.fixed_w
            soft_a = trial_area - sum(
                case.block_area(b) for uu in row + [u] for b in uu.blocks
                if case.fixed_dims(b))
            denom = max(W - trial_fixw, W * 0.15)
            h_implied = max(soft_a / denom if denom > 0 else 0.0, trial_fixh)
            if row and h_implied > 0:
                # close the row when its implied height passes the sweet spot
                target = max(math.sqrt(max(area / max(len(row), 1), 1.0)),
                             fixh, min_h)
                if h_implied > target * 1.6 and area / max(W - fixw, _EPS) >= target * 0.6:
                    break
            row.append(u)
            area = trial_area
            fixh = trial_fixh
            fixw = trial_fixw
            i += 1
        # realize the row
        soft_a = sum(case.block_area(b) for u in row for b in u.blocks
                     if not case.fixed_dims(b))
        if soft_a <= _EPS:
            h = max(fixh, min_h, 1e-6)
        else:
            h = max(soft_a / max(W - fixw, _EPS), fixh, min_h)
        x = x0
        ordered = sorted(row, key=lambda u: (0 if u.boundary & 1 else
                                             (2 if u.boundary & 2 else 1)))
        for u in ordered:
            x = _place_unit_in_row(case, u, x, y, h, out)
        y += h
    return y


def _fill_column(case, blocks, x0, w, y0, y1, obstacles, out, edge_bit):
    """Stack blocks in a vertical column [x0, x0+w) x [y0, y1), skipping
    obstacle y-intervals that intersect the column's x-range. Block widths are
    the column width (heights derive), so every member touches both column
    sides. Returns blocks that did not fit."""
    blocked = []
    for (ox, oy, ow, oh) in obstacles:
        if ox < x0 + w - _EPS and ox + ow > x0 + _EPS:
            blocked.append((oy, oy + oh))
    segs = _free_intervals(y0, y1, blocked)
    rem = list(blocks)
    for (s, e) in segs:
        cap = (e - s) * w
        y = s
        while rem:
            b = rem[0]
            fd = case.fixed_dims(b)
            if fd:
                bw, bh = fd
                if bw > w + 1e-6 or y + bh > e + 1e-6:
                    break
                out[b] = (x0 if edge_bit == 1 else x0 + w - bw, y, bw, bh)
                y += bh
            else:
                bh = case.area[b] / w
                if y + bh > e + 1e-6:
                    break
                out[b] = (x0, y, w, bh)
                y += bh
            rem.pop(0)
    return rem


# ---------------------------------------------------------------------------
# unified obstacle-aware region filler
# ---------------------------------------------------------------------------

def _soft_area(case, u):
    return sum(case.block_area(b) for b in u.blocks
               if not case.fixed_dims(b))


def _unit_ok_at_height(case, u, h, max_aspect=12.0):
    """Would placing u in a row of height h produce absurd slivers?"""
    if u.fixed_h > h + 1e-9:
        return False
    for b in u.blocks:
        if case.fixed_dims(b):
            continue
        w = case.area[b] / h
        if max(w / h, h / w) > max_aspect:
            return False
    return True


def _segment_fill(case, units, segs, y, h, out):
    """Place units into free x-segments of a fixed-height slab [y, y+h).
    Returns units that did not fit (whitespace remainders accepted)."""
    rem = list(units)
    for (a, b) in segs:
        x = a
        j = 0
        while j < len(rem):
            u = rem[j]
            if not _unit_ok_at_height(case, u, h):
                j += 1
                continue
            soft_a = _soft_area(case, u)
            w_need = u.fixed_w + (soft_a / h if soft_a > 0 else 0.0)
            if x + w_need <= b + 1e-9:
                x = _place_unit_in_row(case, u, x, y, h, out)
                rem.pop(j)
            else:
                j += 1
    return rem


def fill_region(case, units, x0, x1, y0, obstacles, out,
                l_queue=None, r_queue=None, xkey=None):
    """Fill the vertical strip [x0, x1) upward from y0, around obstacles.
    Free spans get flexible exact rows; slabs crossed by obstacles get
    segmented fixed-height fills. One unit from l_queue starts each flexible
    row (touching x0) and one from r_queue ends it (touching x1), which is how
    left/right boundary demands are satisfied. Returns the final y."""
    span = x1 - x0
    if span <= 1e-6:
        return y0
    queue = list(units)
    l_queue = l_queue if l_queue is not None else []
    r_queue = r_queue if r_queue is not None else []
    y = y0
    guard = 0
    while (queue or l_queue or r_queue) and guard < 10000:
        guard += 1
        active = [(ox, ox + ow) for (ox, oy, ow, oh) in obstacles
                  if oy <= y + _EPS < oy + oh and ox < x1 - _EPS and ox + ow > x0 + _EPS]
        edges_above = [oy for (ox, oy, ow, oh) in obstacles
                       if oy > y + _EPS and ox < x1 - _EPS and ox + ow > x0 + _EPS]
        edges_above += [oy + oh for (ox, oy, ow, oh) in obstacles
                        if oy + oh > y + _EPS and oy <= y + _EPS
                        and ox < x1 - _EPS and ox + ow > x0 + _EPS]
        if active:
            e = min(edges_above) if edges_above else y + 1.0
            h = e - y
            if h > 1e-6:
                segs = _free_intervals(x0, x1, active)
                queue = _segment_fill(case, queue, segs, y, h, out)
            y = e
            continue
        # free span: flexible exact rows up to the next obstacle edge
        e = min(edges_above) if edges_above else float('inf')
        # build one row: an L unit first, mid units, an R unit last
        row, area, fixh, fixw = [], 0.0, 0.0, 0.0
        def _admit(u):
            nonlocal area, fixh, fixw
            row.append(u)
            area += u.area
            fixh = max(fixh, u.fixed_h)
            fixw += u.fixed_w
        head = tail = None
        if l_queue:
            head = l_queue.pop(0)
            _admit(head)
        if r_queue:
            tail = r_queue.pop(0)
            _admit(tail)
        i = 0
        while i < len(queue):
            u = queue[i]
            t_area = area + u.area
            t_fixh = max(fixh, u.fixed_h)
            t_fixw = fixw + u.fixed_w
            if t_fixw > span * 0.98:
                break
            soft_a = _soft_area_list(case, row + [u])
            denom = max(span - t_fixw, span * 0.15)
            h_impl = max((soft_a / denom) if denom > 0 else 0.0, t_fixh)
            if row:
                target = max(_row_target(case, row, area, span), fixh)
                if h_impl > target * 1.6 and area / max(span - fixw, _EPS) >= target * 0.6:
                    break
            queue.pop(i)
            _admit(u)
        if not row:
            y = e if e != float('inf') else y
            continue
        soft_a = _soft_area_list(case, row)
        h = (max(soft_a / max(span - fixw, _EPS), fixh, 1e-6)
             if soft_a > _EPS else max(fixh, 1e-6))
        if y + h > e - 1e-9:
            # the row would cross into an obstructed slab: clamp its height
            # to land exactly on the edge and let the leftovers re-queue
            h = max(e - y, 1e-6)
            back = []
            x = x0
            for u in row:
                ok = _unit_ok_at_height(case, u, h, max_aspect=25.0)
                w_need = u.fixed_w + (_soft_area(case, u) / h
                                      if _soft_area(case, u) > 0 else 0.0)
                if ok and x + w_need <= x1 + 1e-9:
                    x = _place_unit_in_row(case, u, x, y, h, out)
                elif u is head:
                    l_queue.insert(0, u)
                elif u is tail:
                    r_queue.insert(0, u)
                else:
                    back.append(u)
            queue = back + queue
            y = e
            continue
        x = x0
        mids = [u for u in row if u is not head and u is not tail]
        if xkey is not None:
            mids.sort(key=lambda u: xkey(u, x0, x1))
        for u in ([head] if head else []) + mids:
            x = _place_unit_in_row(case, u, x, y, h, out)
        if tail is not None:
            soft_a = _soft_area(case, tail)
            w_tail = tail.fixed_w + (soft_a / h if soft_a > 0 else 0.0)
            xt = x1 - w_tail
            if xt >= x - 1e-9:
                _place_unit_in_row(case, tail, xt, y, h, out)
            else:
                _place_unit_in_row(case, tail, x, y, h, out)
        y += h
    return y


def _soft_area_list(case, row):
    return sum(_soft_area(case, u) for u in row)


def _row_target(case, row, area, span):
    per = [u.area / max(len(u.blocks), 1) for u in row]
    per.sort()
    return math.sqrt(max(per[len(per) // 2], 1.0))


# ---------------------------------------------------------------------------
# connectivity-aware ordering (interior)
# ---------------------------------------------------------------------------

def order_units(units, b2b, p2b=None, pins=None, H_est=1.0):
    """Vertical ordering by barycenter iteration: each unit is pulled toward
    its connectivity neighbors and (via p2b) toward its pins' absolute y.
    The resulting order fills rows bottom-up, so strongly-connected units and
    pin-tied units land at the right height. Deterministic."""
    m = len(units)
    if m <= 2:
        return list(units)
    idx_of = {}
    for ui, u in enumerate(units):
        for b in u.blocks:
            idx_of[b] = ui
    adj = [dict() for _ in range(m)]
    for a, b, w in b2b:
        ua, ub = idx_of.get(a), idx_of.get(b)
        if ua is None or ub is None or ua == ub:
            continue
        adj[ua][ub] = adj[ua].get(ub, 0.0) + w
        adj[ub][ua] = adj[ub].get(ua, 0.0) + w
    pin_pull = [[0.0, 0.0] for _ in range(m)]   # [weight, weighted-y]
    if p2b and pins is not None:
        for p, b, w in p2b:
            ui = idx_of.get(b)
            if ui is None or p >= len(pins):
                continue
            py = float(pins[p][1])
            pin_pull[ui][0] += w
            pin_pull[ui][1] += w * py
    # initial keys: area-weighted spread (big units early), scaled to H_est
    order0 = sorted(range(m), key=lambda ui: -units[ui].area)
    y = [0.0] * m
    for rank, ui in enumerate(order0):
        y[ui] = H_est * (rank + 0.5) / m
    for _ in range(20):
        ny = list(y)
        for ui in range(m):
            wsum = pin_pull[ui][0]
            acc = pin_pull[ui][1]
            for nb, w in adj[ui].items():
                wsum += w
                acc += w * y[nb]
            if wsum > 0:
                ny[ui] = 0.5 * y[ui] + 0.5 * (acc / wsum)
        y = ny
    order = sorted(range(m), key=lambda ui: (y[ui], -units[ui].area))
    return [units[ui] for ui in order]


def unit_xkey(u, placed_centers, adj_xpull):
    """Within-row x-position estimate for sorting: average pull of already
    placed connected blocks and pins (precomputed per unit)."""
    wsum, acc = adj_xpull.get(id(u), (0.0, 0.0))
    return acc / wsum if wsum > 0 else float('inf')


# ---------------------------------------------------------------------------
# top-level solve
# ---------------------------------------------------------------------------

def dissect_solve(n, areas, b2b_edges, p2b_edges, pins, constraints,
                  target_positions, width_factor=1.0):
    """Frame-of-rows dissection: one full-width bottom row (bottom-required),
    full-width interior rows with left/right-required units injected at the
    row ends, one full-width top row (top-required), everything obstacle-aware
    and exact-fill. Returns positions or None."""
    case = Case(n, areas, constraints, target_positions)
    out: Dict[int, Rect] = {}

    pre = [i for i in range(n) if case.preplaced[i] and case.pre_rect(i)]
    for i in pre:
        out[i] = case.pre_rect(i)
    obstacles = [out[i] for i in pre]
    movable = [i for i in range(n) if i not in set(pre)]
    if not movable:
        return [out.get(i, (0.0, 0.0, 1.0, 1.0)) for i in range(n)]

    units = build_units(case, movable)
    A_mov = sum(u.area for u in units)
    A_pre = sum(r[2] * r[3] for r in obstacles)
    A = A_mov + A_pre

    px1 = max((r[0] + r[2] for r in obstacles), default=0.0)
    py1 = max((r[1] + r[3] for r in obstacles), default=0.0)
    # die sizing: near-square by default, but when obstacles force a tall die
    # (py1 > sqrt(A)) narrow it so the content fills up to the obstacle tops
    # instead of leaving a dead band under the top row
    H_forced = max(math.sqrt(A), py1)
    W = max(min(math.sqrt(A), A / H_forced) * width_factor, px1)

    def route(u):
        c = u.boundary
        if c & 8:
            return 'bottom'
        if c & 4:
            return 'top'
        if c & 1:
            return 'left'
        if c & 2:
            return 'right'
        return 'mid'
    groups = {'bottom': [], 'top': [], 'left': [], 'right': [], 'mid': []}
    for u in units:
        groups[route(u)].append(u)
    for key in ('bottom', 'top'):
        groups[key].sort(key=lambda u: (0 if u.boundary & 1 else
                                        (2 if u.boundary & 2 else 1),
                                        -u.area))
    # order interior for locality; L/R queues largest-first so early (wide)
    # rows take the wide units
    H_est = A / max(W, 1e-9)
    groups['mid'] = order_units(groups['mid'], b2b_edges, p2b_edges, pins,
                                H_est=H_est)
    groups['left'].sort(key=lambda u: -u.area)
    groups['right'].sort(key=lambda u: -u.area)

    # within-row x ordering: pull units toward placed neighbors and pins
    adjb = {}
    for a, b, w in b2b_edges:
        adjb.setdefault(a, []).append((b, w))
        adjb.setdefault(b, []).append((a, w))
    pinx = {}
    if p2b_edges and pins is not None:
        for p, b, w in p2b_edges:
            if 0 <= p < len(pins):
                s = pinx.setdefault(b, [0.0, 0.0])
                s[0] += w
                s[1] += w * float(pins[p][0])

    def xkey(u, xa, xb):
        wsum = acc = 0.0
        for blk in u.blocks:
            for nb, w in adjb.get(blk, ()):
                r = out.get(nb)
                if r is not None:
                    wsum += w
                    acc += w * (r[0] + r[2] / 2)
            s = pinx.get(blk)
            if s:
                wsum += s[0]
                acc += s[1]
        return acc / wsum if wsum > 0 else (xa + xb) / 2

    # --- bottom band: ONE exact row so every member touches y=0 ------------
    y = 0.0
    if groups['bottom']:
        y = _band_row(case, groups['bottom'], W, 0.0, obstacles, out,
                      groups['mid'])

    # --- interior rows with L/R injection ----------------------------------
    y_end = fill_region(case, groups['mid'], 0.0, W, y, obstacles, out,
                        l_queue=groups['left'], r_queue=groups['right'],
                        xkey=xkey)

    # --- top band: ONE exact row flush at the very top ----------------------
    if groups['top']:
        y_top0 = max((r[1] + r[3] for r in out.values()), default=y_end)
        y_top0 = max(y_top0, py1)
        top_spill = []
        y_after = _band_row(case, groups['top'], W, y_top0, [], out, top_spill)
        if top_spill:
            fill_region(case, top_spill, 0.0, W, y_after, [], out)
        _retouch_top(case, groups['top'], out)
    # safety: any block still unplaced goes into a strip above everything
    missing = [i for i in movable if i not in out]
    if missing:
        y_hi = max((r[1] + r[3] for r in out.values()), default=0.0)
        fill_region(case, [Unit([i], 'block', case) for i in missing],
                    0.0, W, y_hi, [], out)
    return [out[i] for i in range(n)]


def _band_row(case, band_units, W, y, obstacles, out, spill):
    """Realize a band as ONE exact-fill row at y spanning [0, W): every soft
    member's height equals the row height, so all touch the band edge.
    Obstacle-crossing segments are handled; units that cannot fit spill into
    the interior queue. Returns the y after the row."""
    fixw = sum(u.fixed_w for u in band_units)
    soft_a = sum(_soft_area(case, u) for u in band_units)
    fixh = max((u.fixed_h for u in band_units), default=0.0)
    h = max(soft_a / max(W - fixw, W * 0.15), fixh, 1e-6)
    blocked = [(ox, ox + ow) for (ox, oy, ow, oh) in obstacles
               if oy < y + h - _EPS and oy + oh > y + _EPS]
    segs = _free_intervals(0.0, W, blocked)
    units = list(band_units)
    right_corner = [u for u in units if u.boundary & 2]
    rest = [u for u in units if not (u.boundary & 2)]
    rem = []
    # reserve space for right-corner units at the very end of the last
    # segment FIRST, so the left-to-right fill can never collide with them
    if right_corner and segs:
        a, b = segs[-1]
        x = b
        placed_rc = []
        for u in reversed(right_corner):
            soft_a = _soft_area(case, u)
            w_need = u.fixed_w + (soft_a / h if soft_a > 0 else 0.0)
            if x - w_need >= a - 1e-9:
                x -= w_need
                placed_rc.append((u, x))
            else:
                rem.append(u)
        for u, xu in placed_rc:
            _place_unit_in_row(case, u, xu, y, h, out)
        segs = segs[:-1] + [(a, x)]
    elif right_corner:
        rem.extend(right_corner)
    rem.extend(_segment_fill(case, rest, segs, y, h, out))
    spill.extend(rem)
    return y + h


def _retouch_top(case, top_units, out):
    """Pull top-required blocks flush to the real y_max when no overlap is
    created."""
    if not top_units:
        return
    y_max = max(r[1] + r[3] for r in out.values())
    for u in top_units:
        for b in u.blocks:
            if not (case.boundary[b] & 4) or b not in out:
                continue
            x, y, w, h = out[b]
            ny = y_max - h
            if ny <= y + 1e-12:
                continue
            clash = False
            for j, (jx, jy, jw, jh) in out.items():
                if j == b:
                    continue
                if (x < jx + jw - 1e-9 and x + w > jx + 1e-9 and
                        ny < jy + jh - 1e-9 and ny + h > jy + 1e-9):
                    clash = True
                    break
            if not clash:
                out[b] = (x, ny, w, h)
