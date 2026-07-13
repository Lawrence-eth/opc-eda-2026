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

    def __init_forced__(self):
        pass

    def fixed_dims(self, i):
        if self.tp is None:
            return None
        w, h = self.tp[i][2], self.tp[i][3]
        return (w, h) if (w != -1 and h != -1) else None

    def rigid_dims(self, i):
        """Dims that must be used verbatim: fixed-shape dims, or dims forced
        for MIB unification (set in dissect_solve for split groups)."""
        fd = self.fixed_dims(i)
        if fd:
            return fd
        forced = getattr(self, 'forced', None)
        if forced:
            return forced.get(i)
        return None

    def pre_rect(self, i):
        if self.tp is None:
            return None
        x, y, w, h = self.tp[i]
        return (x, y, w, h) if -1 not in (x, y, w, h) else None

    def block_area(self, i):
        fd = self.rigid_dims(i)
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
        dims = [case.rigid_dims(b) for b in blocks if case.rigid_dims(b)]
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


def _force_split_mib(case, movable):
    # MIB unification for groups NOT captured as one unit (split across
    # clusters/edge groups): force one shared square shape on every member.
    unit_of = {}
    units0 = build_units(case, movable)
    for u in units0:
        for b in u.blocks:
            unit_of[b] = u
    forced = {}
    mgroups = {}
    for i in movable:
        if case.mib[i] > 0:
            mgroups.setdefault(case.mib[i], []).append(i)
    for grp in mgroups.values():
        if len(grp) < 2:
            continue
        owners = {id(unit_of[b]) for b in grp}
        amin = min(case.area[b] for b in grp)
        amax = max(case.area[b] for b in grp)
        if len(owners) > 1 and amax / amin <= 1.005:
            anyfixed = any(case.fixed_dims(b) for b in grp)
            if not anyfixed:
                s = math.sqrt(amin)
                for b in grp:
                    forced[b] = (s, s)
    case.forced = forced


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

def _place_unit_in_row(case, u, x, y, h, out, flat=False):
    """Place unit u at x in row band [y, y+h). Returns the new x cursor.
    flat=True forces clusters into a single lane (used in the bottom/top
    bands, where every member must keep the band-edge touch)."""
    if u.kind == 'vstack':
        # vertical stack of soft blocks sharing one column: every member
        # spans the full stack width, so all touch the column's x-sides
        w = u.area / h
        yy = y
        for b in u.blocks:
            bh = case.area[b] / w
            out[b] = (x, yy, w, bh)
            yy += bh
        return x + w
    if u.kind == 'mib':
        a = min(case.area[b] for b in u.blocks)
        w = a / h
        for b in u.blocks:
            out[b] = (x, y, w, h)
            x += w
        return x
    if u.kind == 'cluster':
        return _place_cluster(case, u, x, y, h, out, flat=flat)
    b = u.blocks[0]
    fd = case.rigid_dims(b)
    if fd:
        out[b] = (x, y, fd[0], fd[1])
        return x + fd[0]
    w = case.area[b] / h
    out[b] = (x, y, w, h)
    return x + w


def _place_cluster(case, u, x, y, h, out, flat=False):
    blocks = list(u.blocks)
    # members requiring the left/right die edge form vertical stacks on the
    # cluster's outer sides (the cluster itself is routed to that row end),
    # so ALL of them touch the edge, not just the first
    if not flat:
        eL = [b for b in blocks if case.boundary[b] & 1 and not case.rigid_dims(b)]
        eR = [b for b in blocks if case.boundary[b] & 2 and not case.rigid_dims(b)]
        mid = [b for b in blocks if b not in set(eL) | set(eR)]
        if len(eL) > 1 or len(eR) > 1:
            cx = x
            if eL:
                aL = sum(case.area[b] for b in eL)
                wL = aL / h
                yy = y
                for b in eL:
                    bh = case.area[b] / wL
                    out[b] = (cx, yy, wL, bh)
                    yy += bh
                cx += wL
            if mid:
                sub = Unit(mid, 'cluster', case)
                cx = _place_cluster(case, sub, cx, y, h, out, flat=False)
            if eR:
                aR = sum(case.area[b] for b in eR)
                wR = aR / h
                yy = y
                for b in eR:
                    bh = case.area[b] / wR
                    out[b] = (cx, yy, wR, bh)
                    yy += bh
                cx += wR
            return cx
    fixed = [b for b in blocks if case.rigid_dims(b)]
    soft = [b for b in blocks if not case.rigid_dims(b)]
    if flat or fixed or len(blocks) <= 2:
        cx = x
        for b in sorted(fixed, key=lambda b: -case.rigid_dims(b)[1]):
            w, hh = case.rigid_dims(b)
            out[b] = (cx, y, w, hh)
            cx += w
        for b in _edge_order(case, soft):
            w = case.area[b] / h
            out[b] = (cx, y, w, h)
            cx += w
        return cx
    # two stacked lanes of equal width => connected tiling, milder aspects
    blocks.sort(key=lambda b: (0 if case.boundary[b] & 1 else
                               (2 if case.boundary[b] & 2 else 1),
                               -case.area[b]))
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
                     if not case.rigid_dims(b))
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
               if not case.rigid_dims(b))


def _unit_ok_at_height(case, u, h, max_aspect=12.0):
    """Would placing u in a row of height h produce absurd slivers?"""
    if u.fixed_h > h + 1e-9:
        return False
    for b in u.blocks:
        if case.rigid_dims(b):
            continue
        w = case.area[b] / h
        if max(w / h, h / w) > max_aspect:
            return False
    return True


def _segment_fill(case, units, segs, y, h, out, max_aspect=12.0, flat=False):
    """Place units into free x-segments of a fixed-height slab [y, y+h).
    Returns units that did not fit (whitespace remainders accepted)."""
    rem = list(units)
    for (a, b) in segs:
        x = a
        j = 0
        while j < len(rem):
            u = rem[j]
            if not _unit_ok_at_height(case, u, h, max_aspect=max_aspect):
                j += 1
                continue
            soft_a = _soft_area(case, u)
            w_need = u.fixed_w + (soft_a / h if soft_a > 0 else 0.0)
            if x + w_need <= b + 1e-9:
                x = _place_unit_in_row(case, u, x, y, h, out, flat=flat)
                rem.pop(j)
            else:
                j += 1
    return rem


def _trace_append(trace, event, **fields):
    if trace is None:
        return
    row = {"event": event}
    row.update(fields)
    trace.append(row)


def _unit_block_count(units):
    return sum(len(u.blocks) for u in units)


def fill_region(case, units, x0, x1, y0, obstacles, out,
                l_queue=None, r_queue=None, xkey=None,
                trace=None, trace_label="region",
                clamped_backfill=False, active_slab_max_aspect=12.0):
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
    _trace_append(trace, "fill_start", label=trace_label, x0=x0, x1=x1, y=y,
                  units=len(queue), unit_blocks=_unit_block_count(queue),
                  left=len(l_queue), right=len(r_queue),
                  obstacles=len(obstacles))
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
                segs0 = list(segs)
                q_before = len(queue)
                l_before = len(l_queue)
                r_before = len(r_queue)
                before = set(out)
                # die-edge segments can still host L/R-required units
                for si, (a, b) in enumerate(segs):
                    if abs(a - x0) < _EPS and l_queue:
                        u = l_queue[0]
                        if _unit_ok_at_height(case, u, h, 30.0):
                            w_need = u.fixed_w + (_soft_area(case, u) / h
                                                  if _soft_area(case, u) > 0 else 0.0)
                            if a + w_need <= b + 1e-9:
                                l_queue.pop(0)
                                xx = _place_unit_in_row(case, u, a, y, h, out)
                                segs[si] = (xx, b)
                for si, (a, b) in enumerate(segs):
                    if abs(b - x1) < _EPS and r_queue:
                        u = r_queue[0]
                        if _unit_ok_at_height(case, u, h, 30.0):
                            w_need = u.fixed_w + (_soft_area(case, u) / h
                                                  if _soft_area(case, u) > 0 else 0.0)
                            if b - w_need >= a - 1e-9:
                                r_queue.pop(0)
                                _place_unit_in_row(case, u, b - w_need, y, h, out)
                                segs[si] = (a, b - w_need)
                queue = _segment_fill(
                    case, queue, segs, y, h, out,
                    max_aspect=active_slab_max_aspect)
                placed = set(out) - before
                free_area = sum((b - a) for a, b in segs0) * h
                placed_area = sum(out[b][2] * out[b][3] for b in placed)
                _trace_append(
                    trace, "active_slab", label=trace_label, guard=guard,
                    y=y, e=e, h=h, active=len(active), segs=len(segs0),
                    free_area=free_area, placed_blocks=len(placed),
                    placed_area=placed_area,
                    fill_ratio=(placed_area / free_area if free_area > _EPS else 0.0),
                    q_before=q_before, q_after=len(queue),
                    l_before=l_before, l_after=len(l_queue),
                    r_before=r_before, r_after=len(r_queue))
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
        # estimate rows remaining so over-long edge queues stack up now
        q_area = sum(uu.area for uu in queue) or 1.0
        h_typ = max(_row_target(case, queue[:8] or l_queue or r_queue,
                                q_area, span), 1e-6) if (queue or l_queue or r_queue) else 1.0
        rows_left = max(1, round(q_area / (span * h_typ)))
        head = _pop_edge_stack(case, l_queue, rows_left)
        if head is not None:
            _admit(head)
        tail = _pop_edge_stack(case, r_queue, rows_left)
        if tail is not None:
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
            _trace_append(trace, "empty_free_span", label=trace_label,
                          guard=guard, y=y,
                          e=None if e == float('inf') else e,
                          q=len(queue), left=len(l_queue), right=len(r_queue))
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
            before = set(out)
            for u in row:
                if u is head:
                    l_queue.insert(0, u)
                    continue
                if u is tail:
                    r_queue.insert(0, u)
                    continue
                ok = _unit_ok_at_height(case, u, h, max_aspect=25.0)
                w_need = u.fixed_w + (_soft_area(case, u) / h
                                      if _soft_area(case, u) > 0 else 0.0)
                if ok and x + w_need <= x1 + 1e-9:
                    x = _place_unit_in_row(case, u, x, y, h, out)
                else:
                    back.append(u)
            queue = back + queue
            if clamped_backfill and x < x1 - _EPS:
                queue = _segment_fill(
                    case, queue, [(x, x1)], y, h, out, max_aspect=25.0)
            placed = set(out) - before
            placed_area = sum(out[b][2] * out[b][3] for b in placed)
            _trace_append(
                trace, "free_row_clamped", label=trace_label, guard=guard,
                y=y, e=e, h=h, row_units=len(row),
                row_blocks=_unit_block_count(row), placed_blocks=len(placed),
                placed_area=placed_area, span_area=span * h,
                fill_ratio=(placed_area / (span * h) if span * h > _EPS else 0.0),
                q_after=len(queue), left=len(l_queue), right=len(r_queue),
                back_units=len(back))
            y = e
            continue
        before = set(out)
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
        placed = set(out) - before
        placed_area = sum(out[b][2] * out[b][3] for b in placed)
        _trace_append(
            trace, "free_row", label=trace_label, guard=guard, y=y,
            e=None if e == float('inf') else e, h=h,
            row_units=len(row), row_blocks=_unit_block_count(row),
            placed_blocks=len(placed), placed_area=placed_area,
            span_area=span * h,
            fill_ratio=(placed_area / (span * h) if span * h > _EPS else 0.0),
            q_after=len(queue), left=len(l_queue), right=len(r_queue),
            head_blocks=(len(head.blocks) if head else 0),
            tail_blocks=(len(tail.blocks) if tail else 0))
        y += h
    _trace_append(trace, "fill_end", label=trace_label, y=y,
                  guards=guard, units=len(queue), left=len(l_queue),
                  right=len(r_queue))
    return y


def _pop_edge_stack(case, edge_queue, rows_left):
    """Pop 1..k units from an edge queue; if more units remain than rows,
    bundle several SOFT singletons into one vertical stack so they all still
    touch the die edge."""
    if not edge_queue:
        return None
    k = max(1, -(-len(edge_queue) // max(rows_left, 1)))  # ceil
    if len(edge_queue) >= 4:
        k = max(k, 2)
    first = edge_queue.pop(0)
    if k == 1 or first.kind != 'block' or first.fixed_w > 0:
        return first
    stack_blocks = list(first.blocks)
    taken = 1
    i = 0
    while taken < k and i < len(edge_queue):
        u = edge_queue[i]
        if u.kind == 'block' and u.fixed_w == 0:
            stack_blocks.extend(u.blocks)
            edge_queue.pop(i)
            taken += 1
        else:
            i += 1
    if len(stack_blocks) == 1:
        return first
    u = Unit(stack_blocks, 'vstack', case)
    u.boundary = first.boundary
    return u


def _soft_area_list(case, row):
    return sum(_soft_area(case, u) for u in row)


def _row_target(case, row, area, span):
    per = [u.area / max(len(u.blocks), 1) for u in row]
    per.sort()
    return math.sqrt(max(per[len(per) // 2], 1.0))


def _band_snowball_ratio(case, band_units, W, y, obstacles):
    if not band_units or not obstacles or W <= 1e-6:
        return 0.0
    fixw = sum(u.fixed_w for u in band_units)
    soft_a = sum(_soft_area(case, u) for u in band_units)
    fixh = max((u.fixed_h for u in band_units), default=0.0)
    if soft_a <= _EPS:
        return 0.0

    h = max(soft_a / max(W - fixw, W * 0.15), fixh, 1e-6)
    for _ in range(3):
        blocked = [(ox, ox + ow) for (ox, oy, ow, oh) in obstacles
                   if oy < y + h - _EPS and oy + oh > y + _EPS]
        free_w = sum(e - s for s, e in _free_intervals(0.0, W, blocked))
        h2 = max(soft_a / max(free_w - fixw, W * 0.15), fixh, 1e-6)
        if abs(h2 - h) < 1e-9:
            break
        h = h2

    active = [(ox, ox + ow) for (ox, oy, ow, oh) in obstacles
              if oy <= y + _EPS < oy + oh and ox < W - _EPS and ox + ow > _EPS]
    free_w = sum(e - s for s, e in _free_intervals(0.0, W, active))
    h_cap = max(soft_a / max(free_w - fixw, W * 0.15), fixh, 1e-6)
    edges = [oy for (_ox, oy, _ow, _oh) in obstacles if oy > y + _EPS]
    edges += [oy + oh for (_ox, oy, _ow, oh) in obstacles
              if oy <= y + _EPS < oy + oh and oy + oh > y + _EPS]
    if edges:
        h_cap = max(min(h_cap, min(edges) - y), fixh, 1e-6)
    return h / max(h_cap, 1e-6)


def should_try_band_edge_cap(n, areas, constraints, target_positions,
                             width_factor=1.0, min_ratio=5.0):
    """Cheap predictor for the optional obstacle-band cap candidate.

    It fires only when the incumbent bottom band height is much larger than
    the capped height at the first obstacle edge, the case-70/90 failure mode.
    """
    case = Case(n, areas, constraints, target_positions)
    pre = [i for i in range(n) if case.preplaced[i] and case.pre_rect(i)]
    if not pre:
        return False
    out = {i: case.pre_rect(i) for i in pre}
    obstacles = [out[i] for i in pre]
    movable = [i for i in range(n) if i not in set(pre)]
    if not movable:
        return False
    _force_split_mib(case, movable)
    units = build_units(case, movable)
    A_mov = sum(u.area for u in units)
    A_pre = sum(r[2] * r[3] for r in obstacles)
    A = A_mov + A_pre
    px1 = max((r[0] + r[2] for r in obstacles), default=0.0)
    py1 = max((r[1] + r[3] for r in obstacles), default=0.0)
    H_forced = max(math.sqrt(A), py1)
    W = max(min(math.sqrt(A), A / H_forced) * width_factor, px1)
    bottom = [u for u in units if u.boundary & 8]
    return _band_snowball_ratio(case, bottom, W, 0.0, obstacles) >= min_ratio


# ---------------------------------------------------------------------------
# connectivity-aware ordering (interior)
# ---------------------------------------------------------------------------

def order_units(units, b2b, p2b=None, pins=None, H_est=1.0, pin_scale=1.0,
                y_init=None, coordinate_prior=None, prior_weight=0.65):
    """Vertical ordering by barycenter iteration: each unit is pulled toward
    its connectivity neighbors and (via p2b) toward its pins' absolute y.
    On the second pass, b2b edges to blocks outside this queue also pull
    toward those blocks' previous y centers. The resulting order fills rows
    bottom-up, so strongly-connected units and pin-tied units land at the
    right height. Deterministic."""
    m = len(units)
    if m <= 2 and not coordinate_prior:
        return list(units)
    if m <= 1:
        return list(units)
    idx_of = {}
    for ui, u in enumerate(units):
        for b in u.blocks:
            idx_of[b] = ui
    adj = [dict() for _ in range(m)]
    external_y = [[0.0, 0.0] for _ in range(m)]
    for a, b, w in b2b:
        ua, ub = idx_of.get(a), idx_of.get(b)
        if ua is not None and ub is not None and ua != ub:
            adj[ua][ub] = adj[ua].get(ub, 0.0) + w
            adj[ub][ua] = adj[ub].get(ua, 0.0) + w
            continue
        if y_init:
            if ua is not None and ub is None and b in y_init:
                external_y[ua][0] += w
                external_y[ua][1] += w * y_init[b]
            elif ub is not None and ua is None and a in y_init:
                external_y[ub][0] += w
                external_y[ub][1] += w * y_init[a]
    pin_pull = [[0.0, 0.0] for _ in range(m)]   # [weight, weighted-y]
    if p2b and pins is not None:
        for p, b, w in p2b:
            ui = idx_of.get(b)
            if ui is None or p >= len(pins):
                continue
            py = float(pins[p][1])
            pin_pull[ui][0] += w * pin_scale
            pin_pull[ui][1] += w * pin_scale * py
    # initial keys: previous-pass y when available, else area-weighted spread
    order0 = sorted(range(m), key=lambda ui: -units[ui].area)
    y = [0.0] * m
    for rank, ui in enumerate(order0):
        y[ui] = H_est * (rank + 0.5) / m
    if y_init:
        for ui, u in enumerate(units):
            vals = [y_init[b] for b in u.blocks if b in y_init]
            if vals:
                y[ui] = sum(vals) / len(vals)
    for _ in range(20):
        ny = list(y)
        for ui in range(m):
            wsum = pin_pull[ui][0] + external_y[ui][0]
            acc = pin_pull[ui][1] + external_y[ui][1]
            for nb, w in adj[ui].items():
                wsum += w
                acc += w * y[nb]
            if wsum > 0:
                ny[ui] = 0.5 * y[ui] + 0.5 * (acc / wsum)
        y = ny
    order = sorted(range(m), key=lambda ui: (y[ui], -units[ui].area))
    if coordinate_prior:
        # Blend ranks rather than raw coordinates so the learned normalized
        # center and the case-scaled barycenter remain commensurate.  A unit
        # prior is the mean of its MIB/cluster member predictions.
        bary_rank = {ui: rank for rank, ui in enumerate(order)}

        def unit_prior(ui):
            values = [
                coordinate_prior[block][1]
                for block in units[ui].blocks
                if block in coordinate_prior
            ]
            if not values:
                return 0.5
            return sum(values) / len(values)

        learned = sorted(
            range(m), key=lambda ui: (unit_prior(ui), -units[ui].area)
        )
        learned_rank = {ui: rank for rank, ui in enumerate(learned)}
        weight = min(1.0, max(0.0, float(prior_weight)))
        order = sorted(
            range(m),
            key=lambda ui: (
                (1.0 - weight) * bary_rank[ui] + weight * learned_rank[ui],
                -units[ui].area,
            ),
        )
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
                  target_positions, width_factor=1.0, pin_scale=1.0,
                  order_ops=None, trace=None, band_edge_cap=False,
                  edge_order_mode="area", band_order_mode="width",
                  clamped_backfill=False, active_slab_max_aspect=12.0,
                  return_first_pass=False, learned_order=None,
                  learned_prior_weight=0.65):
    """Two-pass frame-of-rows dissection (pass 2 re-orders with pass 1's
    actual positions — one Gauss-Seidel sweep). Returns positions or None."""
    p1 = _dissect_once(n, areas, b2b_edges, p2b_edges, pins, constraints,
                       target_positions, width_factor, pin_scale, order_ops,
                       prev=None, trace=trace, pass_name="p1",
                       band_edge_cap=band_edge_cap,
                       edge_order_mode=edge_order_mode,
                       band_order_mode=band_order_mode,
                       clamped_backfill=clamped_backfill,
                       active_slab_max_aspect=active_slab_max_aspect,
                       learned_order=learned_order,
                       learned_prior_weight=learned_prior_weight)
    if p1 is None:
        return None
    prev = {i: p1[i] for i in range(n)}
    p2 = _dissect_once(n, areas, b2b_edges, p2b_edges, pins, constraints,
                       target_positions, width_factor, pin_scale, order_ops,
                       prev=prev, trace=trace, pass_name="p2",
                       band_edge_cap=band_edge_cap,
                       edge_order_mode=edge_order_mode,
                       band_order_mode=band_order_mode,
                       clamped_backfill=clamped_backfill,
                       active_slab_max_aspect=active_slab_max_aspect,
                       learned_order=learned_order,
                       learned_prior_weight=learned_prior_weight)
    result = p2 if p2 is not None else p1
    if return_first_pass:
        return result, p1
    return result


def _dissect_once(n, areas, b2b_edges, p2b_edges, pins, constraints,
                  target_positions, width_factor=1.0, pin_scale=1.0,
                  order_ops=None, prev=None, trace=None, pass_name="p",
                  band_edge_cap=False, edge_order_mode="area",
                  band_order_mode="width", clamped_backfill=False,
                  active_slab_max_aspect=12.0, learned_order=None,
                  learned_prior_weight=0.65):
    case = Case(n, areas, constraints, target_positions)
    out: Dict[int, Rect] = {}

    pre = [i for i in range(n) if case.preplaced[i] and case.pre_rect(i)]
    for i in pre:
        out[i] = case.pre_rect(i)
    obstacles = [out[i] for i in pre]
    movable = [i for i in range(n) if i not in set(pre)]
    if not movable:
        return [out.get(i, (0.0, 0.0, 1.0, 1.0)) for i in range(n)]

    _force_split_mib(case, movable)
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
    _trace_append(trace, "pass_start", label=pass_name, n=n, width=W,
                  total_area=A, movable_area=A_mov, preplaced_area=A_pre,
                  px1=px1, py1=py1, bottom=len(groups['bottom']),
                  top=len(groups['top']), left=len(groups['left']),
                  right=len(groups['right']), mid=len(groups['mid']),
                  bottom_blocks=_unit_block_count(groups['bottom']),
                  top_blocks=_unit_block_count(groups['top']),
                  left_blocks=_unit_block_count(groups['left']),
                  right_blocks=_unit_block_count(groups['right']),
                  mid_blocks=_unit_block_count(groups['mid']),
                  obstacles=len(obstacles), prev=prev is not None)
    # order interior for locality; L/R queues largest-first so early (wide)
    # rows take the wide units
    H_est = A / max(W, 1e-9)
    y_init = ({b: (prev[b][1] + prev[b][3] / 2) for b in prev}
              if prev else None)
    groups['mid'] = order_units(groups['mid'], b2b_edges, p2b_edges, pins,
                                H_est=H_est, pin_scale=pin_scale,
                                y_init=y_init,
                                coordinate_prior=learned_order,
                                prior_weight=learned_prior_weight)
    if order_ops:
        mid = groups['mid']
        K = len(mid)
        if K > 1:
            for (i, j) in order_ops:
                a, b = i % K, j % K
                mid[a], mid[b] = mid[b], mid[a]
    if edge_order_mode == "bary":
        groups['left'] = order_units(groups['left'], b2b_edges, p2b_edges,
                                     pins, H_est=H_est,
                                     pin_scale=pin_scale, y_init=y_init,
                                     coordinate_prior=learned_order,
                                     prior_weight=learned_prior_weight)
        groups['right'] = order_units(groups['right'], b2b_edges, p2b_edges,
                                      pins, H_est=H_est,
                                      pin_scale=pin_scale, y_init=y_init,
                                      coordinate_prior=learned_order,
                                      prior_weight=learned_prior_weight)
    else:
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
                if r is None and prev is not None:
                    r = prev.get(nb)
                if r is not None:
                    wsum += w
                    acc += w * (r[0] + r[2] / 2)
            s = pinx.get(blk)
            if s:
                wsum += s[0]
                acc += s[1]
        barycenter = acc / wsum if wsum > 0 else (xa + xb) / 2
        if not learned_order:
            return barycenter
        priors = [learned_order[b][0] for b in u.blocks if b in learned_order]
        if not priors:
            return barycenter
        learned_x = W * sum(priors) / len(priors)
        weight = min(1.0, max(0.0, float(learned_prior_weight)))
        return (1.0 - weight) * barycenter + weight * learned_x

    # --- bottom band: ONE exact row so every member touches y=0 ------------
    y = 0.0
    if groups['bottom']:
        y = _band_row(case, groups['bottom'], W, 0.0, obstacles, out,
                      groups['mid'], trace=trace,
                      trace_label=f"{pass_name}:bottom",
                      band_edge_cap=band_edge_cap,
                      band_order_mode=band_order_mode, xkey=xkey)

    # --- interior rows with L/R injection ----------------------------------
    y_end = fill_region(case, groups['mid'], 0.0, W, y, obstacles, out,
                        l_queue=groups['left'], r_queue=groups['right'],
                        xkey=xkey, trace=trace,
                        trace_label=f"{pass_name}:mid",
                        clamped_backfill=clamped_backfill,
                        active_slab_max_aspect=active_slab_max_aspect)

    # --- top band: ONE exact row flush at the very top ----------------------
    if groups['top']:
        y_top0 = max((r[1] + r[3] for r in out.values()), default=y_end)
        y_top0 = max(y_top0, py1)
        top_spill = []
        y_after = _band_row(case, groups['top'], W, y_top0, [], out,
                            top_spill, trace=trace,
                            trace_label=f"{pass_name}:top",
                            band_edge_cap=band_edge_cap,
                            band_order_mode=band_order_mode, xkey=xkey)
        if top_spill:
            fill_region(case, top_spill, 0.0, W, y_after, [], out,
                        trace=trace, trace_label=f"{pass_name}:top_spill")
        _retouch_top(case, groups['top'], out)
    # safety: any block still unplaced goes into a strip above everything
    missing = [i for i in movable if i not in out]
    if missing:
        y_hi = max((r[1] + r[3] for r in out.values()), default=0.0)
        fill_region(case, [Unit([i], 'block', case) for i in missing],
                    0.0, W, y_hi, [], out, trace=trace,
                    trace_label=f"{pass_name}:missing")
    _retouch_edges(case, out, n)
    return [out[i] for i in range(n)]


def _band_row(case, band_units, W, y, obstacles, out, spill,
              trace=None, trace_label="band", band_edge_cap=False,
              band_order_mode="width", xkey=None):
    """Realize a band as ONE exact-fill row at y spanning [0, W): every soft
    member's height equals the row height, so all touch the band edge.
    Obstacle-crossing segments are handled; units that cannot fit spill into
    the interior queue. Returns the y after the row."""
    fixw = sum(u.fixed_w for u in band_units)
    soft_a = sum(_soft_area(case, u) for u in band_units)
    fixh = max((u.fixed_h for u in band_units), default=0.0)
    if obstacles and band_edge_cap:
        # Derive the band height from the x-space available at this y only.
        # If the implied row would cross the next obstacle edge, stop there
        # and spill leftovers. Letting the height feed back through every
        # newly-crossed obstacle can snowball into a very tall mostly-empty
        # band (case 70).
        blocked = [(ox, ox + ow) for (ox, oy, ow, oh) in obstacles
                   if oy <= y + _EPS < oy + oh and ox < W - _EPS and ox + ow > _EPS]
        free_w = sum(e - s for s, e in _free_intervals(0.0, W, blocked))
        h = max(soft_a / max(free_w - fixw, W * 0.15), fixh, 1e-6)
        edges = [oy for (_ox, oy, _ow, _oh) in obstacles if oy > y + _EPS]
        edges += [oy + oh for (_ox, oy, _ow, oh) in obstacles
                  if oy <= y + _EPS < oy + oh and oy + oh > y + _EPS]
        if edges:
            h = max(min(h, min(edges) - y), fixh, 1e-6)
        blocked = [(ox, ox + ow) for (ox, oy, ow, oh) in obstacles
                   if oy < y + h - _EPS and oy + oh > y + _EPS]
    else:
        h = max(soft_a / max(W - fixw, W * 0.15), fixh, 1e-6)
        # obstacles crossing the band eat width: re-derive h from the FREE
        # width. This is the incumbent behavior; keep it as the default and
        # test the capped mode only as an extra best-of candidate.
        for _ in range(3):
            blocked = [(ox, ox + ow) for (ox, oy, ow, oh) in obstacles
                       if oy < y + h - _EPS and oy + oh > y + _EPS]
            free_w = sum(e - s for s, e in _free_intervals(0.0, W, blocked))
            h2 = max(soft_a / max(free_w - fixw, W * 0.15), fixh, 1e-6)
            if abs(h2 - h) < 1e-9:
                break
            h = h2
    segs = _free_intervals(0.0, W, blocked)
    units = list(band_units)
    right_corner = [u for u in units if u.boundary & 2]
    rest = [u for u in units if not (u.boundary & 2)]
    if band_order_mode == "pinx" and xkey is not None:
        right_corner.sort(key=lambda u: (xkey(u, 0.0, W), -u.area))
    rem = []
    before = set(out)
    # NOTE: slivers are legal (evaluator checks area only); bands use a very
    # loose aspect guard so small blocks still make it onto their edge
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
            _place_unit_in_row(case, u, xu, y, h, out, flat=True)
        segs = segs[:-1] + [(a, x)]
    elif right_corner:
        rem.extend(right_corner)
    if band_order_mode == "pinx" and xkey is not None:
        rest = sorted(rest, key=lambda u: (xkey(u, 0.0, W),
                                           -(u.fixed_w + (_soft_area(case, u) / h
                                             if _soft_area(case, u) > 0 else 0.0))))
    else:
        rest = sorted(rest, key=lambda u: -(u.fixed_w + (_soft_area(case, u) / h
                                            if _soft_area(case, u) > 0 else 0.0)))
    rem.extend(_segment_fill(case, rest, segs, y, h, out, max_aspect=60.0,
                             flat=True))
    spill.extend(rem)
    placed = set(out) - before
    placed_area = sum(out[b][2] * out[b][3] for b in placed)
    free_area = sum((b - a) for a, b in segs) * h
    _trace_append(trace, "band_row", label=trace_label, y=y, h=h,
                  units=len(band_units), unit_blocks=_unit_block_count(band_units),
                  placed_blocks=len(placed), placed_area=placed_area,
                  free_area=free_area,
                  fill_ratio=(placed_area / free_area if free_area > _EPS else 0.0),
                  spill=len(rem), obstacles=len(blocked))
    return y + h


def _retouch_edges(case, out, n):
    """Final pass: for every boundary-coded block not on its edge, move it
    flush if the destination space is empty (checked live). Positions only
    ever move OUTWARD toward the current bbox edge, so the bbox is stable."""
    def bbox():
        xm = min(r[0] for r in out.values()); ym = min(r[1] for r in out.values())
        xM = max(r[0] + r[2] for r in out.values()); yM = max(r[1] + r[3] for r in out.values())
        return xm, ym, xM, yM
    for _ in range(2):
        xm, ym, xM, yM = bbox()
        for i in range(n):
            c = case.boundary[i]
            if not c or i not in out:
                continue
            if case.preplaced[i]:
                # NEVER move a preplaced block toward a soft boundary edge —
                # preplaced position is a hard constraint (Q&A Q5)
                continue
            x, y, w, h = out[i]
            nx, ny = x, y
            if c & 1 and abs(x - xm) > 1e-9:
                nx = xm
            if c & 2 and abs(x + w - xM) > 1e-9:
                nx = xM - w
            if c & 8 and abs(y - ym) > 1e-9:
                ny = ym
            if c & 4 and abs(y + h - yM) > 1e-9:
                ny = yM - h
            if nx == x and ny == y:
                continue
            clash = False
            for j, (jx, jy, jw, jh) in out.items():
                if j == i:
                    continue
                if (nx < jx + jw - 1e-9 and nx + w > jx + 1e-9 and
                        ny < jy + jh - 1e-9 and ny + h > jy + 1e-9):
                    clash = True
                    break
            if not clash:
                out[i] = (nx, ny, w, h)


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
