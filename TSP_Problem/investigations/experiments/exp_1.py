import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Data generation
# ---------------------------

def gen_euclidean_points(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2), dtype=float)


def gen_grid_points(m: int) -> np.ndarray:
    # m x m grid => n = m^2
    xs, ys = np.meshgrid(np.arange(m, dtype=float), np.arange(m, dtype=float))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    return pts


def euclidean_dist_matrix(points: np.ndarray) -> np.ndarray:
    # O(n^2) full matrix (fine for n up to a few thousand in Python)
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


# ---------------------------
# Turn penalty p(i, j, k)
# ---------------------------

def angle_turn_penalty(points: np.ndarray, i: int, j: int, k: int) -> float:
    """
    p(i,j,k) = turning angle at vertex j when moving i->j->k.
    If i or k is None (not used here) then p=0.
    """
    if i == j or j == k or i == k:
        return 0.0
    a = points[i] - points[j]  # incoming vector (from j to i)
    b = points[k] - points[j]  # outgoing vector (from j to k)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    cosv = float(np.dot(a, b) / (na * nb))
    cosv = max(-1.0, min(1.0, cosv))
    return float(math.acos(cosv))  # in radians, [0, pi]


# Optional: "step" penalty that is easier to interpret (uncomment to use)
# def step_turn_penalty(points: np.ndarray, i: int, j: int, k: int) -> float:
#     theta = angle_turn_penalty(points, i, j, k)
#     # 0 for ~straight, 1 for ~90deg, 2 for ~U-turn
#     if theta < math.pi * 0.25:
#         return 0.0
#     elif theta < math.pi * 0.75:
#         return 1.0
#     else:
#         return 2.0


# ---------------------------
# Tour utilities
# ---------------------------

def tour_length(d: np.ndarray, tour: List[int]) -> float:
    n = len(tour)
    s = 0.0
    for t in range(n):
        a = tour[t]
        b = tour[(t + 1) % n]
        s += float(d[a, b])
    return s


def tour_turn_sum(points: np.ndarray, tour: List[int]) -> float:
    n = len(tour)
    s = 0.0
    for t in range(n):
        i = tour[(t - 1) % n]
        j = tour[t]
        k = tour[(t + 1) % n]
        s += angle_turn_penalty(points, i, j, k)
    return s


def total_cost(points: np.ndarray, d: np.ndarray, tour: List[int], lam: float) -> Tuple[float, float, float]:
    base = tour_length(d, tour)
    turns = tour_turn_sum(points, tour)
    return base + lam * turns, base, turns


def nearest_neighbor_tour(d: np.ndarray, start: int = 0) -> List[int]:
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        # choose nearest unvisited by distance only
        nxt = int(np.argmin(np.where(visited, np.inf, d[cur])))
        tour.append(nxt)
        visited[nxt] = True
        cur = nxt
    return tour


def second_order_greedy_tour(points: np.ndarray, d: np.ndarray, lam: float, start: int = 0) -> List[int]:
    """
    Turn-aware greedy:
    - pick 2nd vertex by distance
    - then pick next k minimizing d(j,k) + lam*p(i,j,k)
    """
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True

    # choose second vertex by distance (no turn cost yet)
    j = start
    k = int(np.argmin(np.where(visited, np.inf, d[j])))
    tour.append(k)
    visited[k] = True

    prev = start
    cur = k
    for _ in range(n - 2):
        best = None
        best_score = float("inf")
        for cand in range(n):
            if visited[cand]:
                continue
            score = float(d[cur, cand]) + lam * angle_turn_penalty(points, prev, cur, cand)
            if score < best_score:
                best_score = score
                best = cand
        tour.append(int(best))
        visited[best] = True
        prev, cur = cur, best
    return tour


def build_candidates(d: np.ndarray, K: int) -> List[List[int]]:
    """
    Candidate list Cand(i): K nearest neighbors by distance.
    """
    n = d.shape[0]
    cands = []
    for i in range(n):
        idx = np.argsort(d[i])
        # exclude itself (first is i with distance 0)
        neigh = [int(x) for x in idx[1:K+1]]
        cands.append(neigh)
    return cands


# ---------------------------
# 2-opt (distance-only) and 2-opt-turn-aware
# ---------------------------

def two_opt_distance(d: np.ndarray, tour: List[int], cand: Optional[List[List[int]]] = None,
                     max_passes: int = 50) -> List[int]:
    """
    Classic 2-opt on distance.
    If cand provided: only consider edges (a,c) where c in Cand(a).
    """
    n = len(tour)
    pos = np.empty(n, dtype=int)
    for i, v in enumerate(tour):
        pos[v] = i

    def idx_of(v: int) -> int:
        return int(pos[v])

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1

        for ai in range(n):
            a = tour[ai]
            b = tour[(ai + 1) % n]

            # restrict c by candidates if given
            c_list = cand[a] if cand is not None else None
            if c_list is None:
                # full scan of c is O(n^2) — ok for small n, but slower
                c_iter = range(n)
                # we will interpret ci as index, but need vertex
                for ci in range(n):
                    if ci == ai or ci == (ai + 1) % n:
                        continue
                    c = tour[ci]
                    di = (ci + 1) % n
                    dV = tour[di]
                    if dV == a:
                        continue
                    delta = float(d[a, c] + d[b, dV] - d[a, b] - d[c, dV])
                    if delta < -1e-12:
                        # reverse segment (ai+1 .. ci)
                        i1 = (ai + 1) % n
                        i2 = ci
                        if i1 < i2:
                            tour[i1:i2+1] = reversed(tour[i1:i2+1])
                        else:
                            seg = (tour[i1:] + tour[:i2+1])[::-1]
                            tour[i1:] = seg[:n - i1]
                            tour[:i2+1] = seg[n - i1:]
                        for idx, v in enumerate(tour):
                            pos[v] = idx
                        improved = True
                        break
                if improved:
                    break
            else:
                # iterate candidate vertices for c
                for c in c_list:
                    ci = idx_of(c)
                    if ci == ai or ci == (ai + 1) % n:
                        continue
                    di = (ci + 1) % n
                    dV = tour[di]
                    if dV == a:
                        continue
                    delta = float(d[a, c] + d[b, dV] - d[a, b] - d[c, dV])
                    if delta < -1e-12:
                        i1 = (ai + 1) % n
                        i2 = ci
                        if i1 < i2:
                            tour[i1:i2+1] = reversed(tour[i1:i2+1])
                        else:
                            seg = (tour[i1:] + tour[:i2+1])[::-1]
                            tour[i1:] = seg[:n - i1]
                            tour[:i2+1] = seg[n - i1:]
                        for idx, v in enumerate(tour):
                            pos[v] = idx
                        improved = True
                        break
                if improved:
                    break

    return tour


def local_turn_delta_for_positions(points: np.ndarray, tour: List[int], idxs: List[int], lam: float) -> float:
    """
    Recompute turn penalties for a set of positions (indices in tour),
    return sum_{t in idxs} lam * p(prev, cur, next).
    """
    n = len(tour)
    s = 0.0
    for t in idxs:
        i = tour[(t - 1) % n]
        j = tour[t]
        k = tour[(t + 1) % n]
        s += lam * angle_turn_penalty(points, i, j, k)
    return s


def two_opt_turn_aware(points: np.ndarray, d: np.ndarray, tour: List[int], lam: float,
                      cand: Optional[List[List[int]]] = None, max_passes: int = 50) -> List[int]:
    """
    2-opt where improvement is computed on full objective:
      F = sum d + lam * sum p(i,j,k)
    Uses local recomputation of turn penalties around affected indices.

    Note: For correctness, we accept only if deltaF < 0 => monotone decrease of F.
    """
    n = len(tour)
    pos = np.empty(n, dtype=int)
    for i, v in enumerate(tour):
        pos[v] = i

    def idx_of(v: int) -> int:
        return int(pos[v])

    def apply_2opt(ai: int, ci: int):
        i1 = (ai + 1) % n
        i2 = ci
        if i1 < i2:
            tour[i1:i2+1] = reversed(tour[i1:i2+1])
        else:
            seg = (tour[i1:] + tour[:i2+1])[::-1]
            tour[i1:] = seg[:n - i1]
            tour[:i2+1] = seg[n - i1:]
        for idx, v in enumerate(tour):
            pos[v] = idx

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1

        for ai in range(n):
            a = tour[ai]
            bi = (ai + 1) % n
            b = tour[bi]

            # candidate c selection
            c_vertices = cand[a] if cand is not None else None
            if c_vertices is None:
                iter_cs = (tour[ci] for ci in range(n))
            else:
                iter_cs = (c for c in c_vertices)

            for c in iter_cs:
                ci = idx_of(c)
                if ci == ai or ci == bi:
                    continue
                di = (ci + 1) % n
                dV = tour[di]
                if dV == a:
                    continue

                # Distance delta for replacing (a,b)+(c,d) -> (a,c)+(b,d)
                delta_d = float(d[a, c] + d[b, dV] - d[a, b] - d[c, dV])

                # Turn part: recompute locally around affected vertices.
                # Affected positions: ai, bi, ci, di and their neighbors (safe superset).
                affected = set()
                for t in [ai, bi, ci, di]:
                    for u in [t - 1, t, t + 1]:
                        affected.add(u % n)
                affected = sorted(affected)

                old_turn = local_turn_delta_for_positions(points, tour, affected, lam)

                # apply move
                apply_2opt(ai, ci)

                new_turn = local_turn_delta_for_positions(points, tour, affected, lam)

                deltaF = delta_d + (new_turn - old_turn)

                if deltaF < -1e-12:
                    improved = True
                    break  # keep move
                else:
                    # rollback by applying same 2opt again
                    apply_2opt(ai, ci)

            if improved:
                break

    return tour


# ---------------------------
# Experiment runner
# ---------------------------

@dataclass
class ResultRow:
    name: str
    n: int
    lam: float
    algo: str
    F: float
    D: float
    P: float
    secs: float


def run_one_instance(name: str, points: np.ndarray, lam_list: List[float],
                     K_candidates: Optional[int] = None, seed_start: int = 0) -> List[ResultRow]:
    d = euclidean_dist_matrix(points)
    n = d.shape[0]
    cand = build_candidates(d, K_candidates) if K_candidates is not None else None

    rows: List[ResultRow] = []

    for lam in lam_list:
        # --- Algorithm A: distance-based NN + 2-opt (distance only), then evaluate full F ---
        t0 = time.time()
        tourA = nearest_neighbor_tour(d, start=seed_start % n)
        tourA = two_opt_distance(d, tourA, cand=cand, max_passes=30)
        F_A, D_A, P_A = total_cost(points, d, tourA, lam)
        tA = time.time() - t0
        rows.append(ResultRow(name, n, lam, "Baseline(NN+2opt on d)", F_A, D_A, P_A, tA))

        # --- Algorithm B: turn-aware greedy + 2-opt-turn-aware ---
        t0 = time.time()
        tourB = second_order_greedy_tour(points, d, lam, start=seed_start % n)
        tourB = two_opt_turn_aware(points, d, tourB, lam, cand=cand, max_passes=30)
        F_B, D_B, P_B = total_cost(points, d, tourB, lam)
        tB = time.time() - t0
        rows.append(ResultRow(name, n, lam, "TurnAware(Greedy2nd+2opt)", F_B, D_B, P_B, tB))

    return rows


def summarize(rows: List[ResultRow]) -> None:
    # Print a compact table
    print(f"{'Instance':<14} {'n':>6} {'lam':>6} {'algo':<28} {'F':>12} {'D':>12} {'P':>12} {'sec':>8}")
    for r in rows:
        print(f"{r.name:<14} {r.n:>6} {r.lam:>6.2f} {r.algo:<28} {r.F:>12.4f} {r.D:>12.4f} {r.P:>12.4f} {r.secs:>8.3f}")


def plot_improvement(rows: List[ResultRow], title: str = "Improvement vs lambda") -> None:
    # For each instance, plot % improvement of TurnAware over Baseline
    # improvement = (F_baseline - F_turnaware) / F_baseline * 100
    by_key: Dict[Tuple[str, float], Dict[str, float]] = {}
    for r in rows:
        key = (r.name, r.lam)
        by_key.setdefault(key, {})
        by_key[key][r.algo] = r.F

    inst_names = sorted(set(r.name for r in rows))
    lam_list = sorted(set(r.lam for r in rows))

    plt.figure()
    for inst in inst_names:
        xs = []
        ys = []
        for lam in lam_list:
            key = (inst, lam)
            if key not in by_key:
                continue
            fA = by_key[key].get("Baseline(NN+2opt on d)")
            fB = by_key[key].get("TurnAware(Greedy2nd+2opt)")
            if fA is None or fB is None:
                continue
            xs.append(lam)
            ys.append((fA - fB) / fA * 100.0)
        if xs:
            plt.plot(xs, ys, marker="o", label=inst)

    plt.xlabel("lambda")
    plt.ylabel("Improvement in F (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main (minimal experiment)
# ---------------------------

if __name__ == "__main__":
    # Minimal set for a "review-proof" experiment:
    lam_list = [0.0, 0.1, 0.5, 1.0, 2.0]

    all_rows: List[ResultRow] = []

    # Euclidean random instances
    for n in [100, 300, 800]:
        pts = gen_euclidean_points(n, seed=42 + n)
        rows = run_one_instance(f"Rand2D_{n}", pts, lam_list, K_candidates=15, seed_start=0)
        all_rows.extend(rows)

    # Grid instance (road-like)
    pts_grid = gen_grid_points(25)  # 625 nodes
    rows = run_one_instance("Grid_25x25", pts_grid, lam_list, K_candidates=8, seed_start=0)
    all_rows.extend(rows)

    summarize(all_rows)
    plot_improvement(all_rows, title="Turn-aware vs Baseline: Improvement in F")