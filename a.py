from __future__ import annotations

from math import log2
import time
import random
from statistics import mean, stdev

from dataclasses import dataclass
from functools import cmp_to_key
from collections import deque, defaultdict
from typing import Iterable, Optional, Callable


# -----------------------------
# Stats and comparison oracle
# -----------------------------

@dataclass
class SortStats:
    output: list[int]
    algorithm: str
    n: int
    m_known_edges: int
    total_item_comparisons: int
    additional_item_comparisons: int
    known_or_implied_comparisons_used: int
    heap_inserts: int = 0
    heap_delete_mins: int = 0
    heap_merges: int = 0


class ComparisonOracle:
    """
    Compares distinct integer items.

    - total_item_comparisons counts every item comparison requested by the algorithm.
    - additional_item_comparisons counts comparisons not already known/implied by the
      supplied known comparison DAG, when use_transitive_known=True.
    """

    def __init__(
        self,
        vertices: Iterable[int],
        known_less: Iterable[tuple[int, int]] = (),
        *,
        use_transitive_known: bool = True,
    ):
        self.vertices = list(vertices)
        self.vertex_set = set(self.vertices)
        self.total_item_comparisons = 0
        self.additional_item_comparisons = 0
        self.known_or_implied_comparisons_used = 0

        self.direct_adj: dict[int, set[int]] = {v: set() for v in self.vertices}
        for a, b in known_less:
            if a not in self.vertex_set or b not in self.vertex_set:
                raise ValueError(f"Known comparison ({a}, {b}) mentions item not in input.")
            if a == b:
                raise ValueError("Known comparisons must be between distinct items.")
            if not (a < b):
                raise ValueError(f"Known comparison ({a} < {b}) contradicts integer order.")
            self.direct_adj[a].add(b)

        self.reachable: Optional[dict[int, set[int]]] = None
        if use_transitive_known:
            self.reachable = self._compute_reachability()

    def _compute_reachability(self) -> dict[int, set[int]]:
        reachable: dict[int, set[int]] = {v: set() for v in self.vertices}
        for s in self.vertices:
            seen = set()
            stack = list(self.direct_adj[s])
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(self.direct_adj[x] - seen)
            reachable[s] = seen
        return reachable

    def _already_known(self, a: int, b: int) -> bool:
        if a == b:
            return True

        if self.reachable is not None:
            return b in self.reachable[a] or a in self.reachable[b]

        return b in self.direct_adj[a] or a in self.direct_adj[b]

    def compare(self, a: int, b: int) -> int:
        """
        Returns -1 if a < b, +1 if a > b, and 0 if equal.
        For this use case, items should be distinct.
        """
        self.total_item_comparisons += 1

        if self._already_known(a, b):
            self.known_or_implied_comparisons_used += 1
        else:
            self.additional_item_comparisons += 1

        return (a > b) - (a < b)


# -----------------------------
# Pairing heap
# -----------------------------

class _PairingNode:
    __slots__ = ("key", "child", "sibling")

    def __init__(self, key: int):
        self.key = key
        self.child: Optional[_PairingNode] = None
        self.sibling: Optional[_PairingNode] = None


class PairingHeap:
    """
    Min pairing heap with insert, find_min, delete_min.
    The heap delegates all item comparisons to cmp(a, b).
    """

    def __init__(self, cmp: Callable[[int, int], int]):
        self.root: Optional[_PairingNode] = None
        self.cmp = cmp
        self.size = 0
        self.inserts = 0
        self.delete_mins = 0
        self.merges = 0

    def __bool__(self) -> bool:
        return self.root is not None

    def _merge(
        self,
        a: Optional[_PairingNode],
        b: Optional[_PairingNode],
    ) -> Optional[_PairingNode]:
        if a is None:
            return b
        if b is None:
            return a

        self.merges += 1
        if self.cmp(b.key, a.key) < 0:
            a.sibling = b.child
            b.child = a
            return b
        else:
            b.sibling = a.child
            a.child = b
            return a

    def insert(self, key: int) -> None:
        self.inserts += 1
        self.size += 1
        self.root = self._merge(self.root, _PairingNode(key))

    def find_min(self) -> int:
        if self.root is None:
            raise IndexError("find_min from empty heap")
        return self.root.key

    def delete_min(self) -> int:
        if self.root is None:
            raise IndexError("delete_min from empty heap")

        self.delete_mins += 1
        self.size -= 1

        min_key = self.root.key
        first_child = self.root.child
        self.root = self._merge_pairs(first_child)
        return min_key

    def _merge_pairs(self, first: Optional[_PairingNode]) -> Optional[_PairingNode]:
        if first is None:
            return None

        merged_pairs: list[_PairingNode] = []
        cur = first

        # First pass: merge siblings left-to-right in pairs.
        while cur is not None:
            a = cur
            b = cur.sibling
            next_pair = None if b is None else b.sibling

            a.sibling = None
            if b is not None:
                b.sibling = None
                merged_pairs.append(self._merge(a, b))
            else:
                merged_pairs.append(a)

            cur = next_pair

        # Second pass: merge right-to-left.
        root = None
        for node in reversed(merged_pairs):
            root = self._merge(root, node)

        return root


# -----------------------------
# DAG helpers
# -----------------------------

def _build_graph(
    items: Iterable[int],
    known_less: Iterable[tuple[int, int]],
) -> tuple[list[int], dict[int, set[int]], dict[int, int]]:
    vertices = list(items)
    if len(vertices) != len(set(vertices)):
        raise ValueError("This implementation assumes distinct integer items.")

    vertex_set = set(vertices)
    adj = {v: set() for v in vertices}
    indeg = {v: 0 for v in vertices}

    for a, b in known_less:
        if a not in vertex_set or b not in vertex_set:
            raise ValueError(f"Known comparison ({a}, {b}) mentions item not in input.")
        if a == b:
            raise ValueError("Known comparisons must be between distinct items.")
        if not (a < b):
            raise ValueError(f"Known comparison ({a} < {b}) contradicts integer order.")
        if b not in adj[a]:
            adj[a].add(b)
            indeg[b] += 1

    return vertices, adj, indeg


def _topological_levels(
    vertices: list[int],
    adj: dict[int, set[int]],
    indeg: dict[int, int],
) -> dict[int, int]:
    q = deque([v for v in vertices if indeg[v] == 0])
    indeg_work = dict(indeg)
    level = {v: 1 for v in vertices}
    processed = 0

    while q:
        v = q.popleft()
        processed += 1
        for w in adj[v]:
            if level[w] < level[v] + 1:
                level[w] = level[v] + 1
            indeg_work[w] -= 1
            if indeg_work[w] == 0:
                q.append(w)

    if processed != len(vertices):
        raise ValueError("Known comparisons contain a cycle; not a valid partial order.")

    return level


def _make_stats(
    output: list[int],
    algorithm: str,
    n: int,
    m: int,
    oracle: ComparisonOracle,
    heap: Optional[PairingHeap] = None,
) -> SortStats:
    return SortStats(
        output=output,
        algorithm=algorithm,
        n=n,
        m_known_edges=m,
        total_item_comparisons=oracle.total_item_comparisons,
        additional_item_comparisons=oracle.additional_item_comparisons,
        known_or_implied_comparisons_used=oracle.known_or_implied_comparisons_used,
        heap_inserts=0 if heap is None else heap.inserts,
        heap_delete_mins=0 if heap is None else heap.delete_mins,
        heap_merges=0 if heap is None else heap.merges,
    )


# -----------------------------
# Algorithm 1: topological heapsort
# -----------------------------

def topological_heapsort(
    items: Iterable[int],
    known_less: Iterable[tuple[int, int]] = (),
    *,
    use_transitive_known_for_stats: bool = True,
) -> SortStats:
    """
    Basic topological heapsort.

    items:
        Distinct integers to sort.

    known_less:
        Iterable of already-known comparisons (a, b), meaning a < b.
        These become arcs a -> b in the DAG.

    use_transitive_known_for_stats:
        If True, a comparison implied by a known path is not counted as an
        additional comparison in the returned stats.
    """
    known_less = list(known_less)
    vertices, adj, indeg = _build_graph(items, known_less)

    # Validates acyclicity.
    _topological_levels(vertices, adj, indeg)

    oracle = ComparisonOracle(
        vertices,
        known_less,
        use_transitive_known=use_transitive_known_for_stats,
    )
    heap = PairingHeap(oracle.compare)

    indeg_work = dict(indeg)
    for v in vertices:
        if indeg_work[v] == 0:
            heap.insert(v)

    output: list[int] = []

    while len(output) < len(vertices):
        if not heap:
            raise ValueError("No source available; known comparisons contain a cycle.")

        v = heap.delete_min()
        output.append(v)

        for w in adj[v]:
            indeg_work[w] -= 1
            if indeg_work[w] == 0:
                heap.insert(w)

    return _make_stats(
        output,
        "topological_heapsort",
        len(vertices),
        len(known_less),
        oracle,
        heap,
    )


# -----------------------------
# Algorithm 2: topological heapsort with lookahead
# -----------------------------

def topological_heapsort_with_lookahead(
    items: Iterable[int],
    known_less: Iterable[tuple[int, int]] = (),
    *,
    use_transitive_known_for_stats: bool = True,
) -> SortStats:
    """
    Topological heapsort with lookahead, following Section 5.

    Bottlenecks are vertices that are alone on their longest-path level.
    They are kept out of the heap and handled through the lookahead array B.
    """
    known_less = list(known_less)
    vertices, adj, indeg = _build_graph(items, known_less)

    level = _topological_levels(vertices, adj, indeg)

    by_level: dict[int, list[int]] = defaultdict(list)
    for v in vertices:
        by_level[level[v]].append(v)

    bottleneck_set = {
        vs[0]
        for vs in by_level.values()
        if len(vs) == 1
    }
    non_bottleneck_set = set(vertices) - bottleneck_set

    bottlenecks = sorted(bottleneck_set, key=lambda x: level[x])

    # Mark bottlenecks. For each non-bottleneck w, mark the bottleneck v of
    # highest level such that v -> w is a known arc.
    marked_bottlenecks: set[int] = set()
    marked_non_bottlenecks: set[int] = set()

    for v in sorted(bottlenecks, key=lambda x: level[x], reverse=True):
        for w in adj[v]:
            if w in non_bottleneck_set and w not in marked_non_bottlenecks:
                marked_bottlenecks.add(v)
                marked_non_bottlenecks.add(w)

    oracle = ComparisonOracle(
        vertices,
        known_less,
        use_transitive_known=use_transitive_known_for_stats,
    )
    heap = PairingHeap(oracle.compare)

    indeg_work = dict(indeg)
    deleted: set[int] = set()
    output: list[int] = []

    for v in vertices:
        if indeg_work[v] == 0 and v not in bottleneck_set:
            heap.insert(v)

    # B is represented as a list prefix of remaining bottlenecks.
    next_bottleneck_idx = 0
    B: list[int] = []

    def refill_B() -> None:
        nonlocal next_bottleneck_idx, B

        B = []
        while next_bottleneck_idx < len(bottlenecks):
            u = bottlenecks[next_bottleneck_idx]
            next_bottleneck_idx += 1
            if u in deleted:
                continue

            B.append(u)
            if u in marked_bottlenecks:
                break

    refill_B()

    def delete_from_graph(v: int) -> None:
        if v in deleted:
            return
        deleted.add(v)
        output.append(v)

        for w in adj[v]:
            indeg_work[w] -= 1
            if indeg_work[w] == 0 and w not in bottleneck_set:
                heap.insert(w)

    def largest_B_index_less_than(v: int) -> int:
        """
        Exponential search followed by binary search.
        Returns largest j such that B[j] < v.
        Requires B[0] < v < B[-1] or B[0] < v <= B[-1].
        """
        if not B or oracle.compare(B[0], v) >= 0:
            return -1

        # Exponential phase: compare B[1], B[2], B[4], ...
        lo = 0
        idx = 1
        while idx < len(B) and oracle.compare(B[idx], v) < 0:
            lo = idx
            idx *= 2

        hi = min(idx, len(B) - 1)

        # Binary phase over (lo, hi].
        ans = lo
        left, right = lo + 1, hi
        while left <= right:
            mid = (left + right) // 2
            if oracle.compare(B[mid], v) < 0:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1

        return ans

    while len(output) < len(vertices):
        if not heap and not B:
            raise ValueError("No source available; known comparisons contain a cycle or lookahead invariant failed.")

        # Case 1:
        # H non-empty, and either B empty or min-level vertex in B is greater than find_min(H).
        if heap and (not B or oracle.compare(heap.find_min(), B[0]) < 0):
            v = heap.delete_min()
            delete_from_graph(v)
            continue

        # Case 2:
        # B non-empty, and either H empty or find_min(H) is larger than max-level vertex in B.
        if B and (not heap or oracle.compare(B[-1], heap.find_min()) < 0):
            to_process = B
            B = []
            for u in to_process:
                delete_from_graph(u)
            refill_B()
            continue

        # Case 3:
        # H and B are both non-empty, and find_min(H) lies inside the B range.
        v = heap.find_min()
        j = largest_B_index_less_than(v)

        if j < 0:
            # This should not happen for distinct integers if cases are implemented correctly.
            # Fall back to deleting heap minimum to avoid an infinite loop.
            u = heap.delete_min()
            delete_from_graph(u)
            continue

        to_process = B[:j + 1]
        B = B[j + 1:]
        for u in to_process:
            delete_from_graph(u)

    return _make_stats(
        output,
        "topological_heapsort_with_lookahead",
        len(vertices),
        len(known_less),
        oracle,
        heap,
    )


# -----------------------------
# Baseline: normal Python sort with counted comparisons
# -----------------------------

def sort_from_scratch_counted(items: Iterable[int]) -> SortStats:
    """
    Baseline: ignore all known comparisons and sort with Python's comparison sort.
    Counts comparator calls made by Python's Timsort.

    Note: Timsort is adaptive, so counts depend on the input order.
    """
    vertices = list(items)
    oracle = ComparisonOracle(vertices, known_less=(), use_transitive_known=False)

    out = sorted(vertices, key=cmp_to_key(oracle.compare))

    return _make_stats(
        out,
        "python_sorted_from_scratch",
        len(vertices),
        0,
        oracle,
        heap=None,
    )


# -----------------------------
# Convenience benchmark wrapper
# -----------------------------

def compare_all(
    items: Iterable[int],
    known_less: Iterable[tuple[int, int]] = (),
    *,
    use_transitive_known_for_stats: bool = True,
) -> dict[str, SortStats]:
    """
    Runs:
      1. Python sorted from scratch.
      2. Topological heapsort.
      3. Topological heapsort with lookahead.
    """
    items = list(items)
    known_less = list(known_less)

    return {
        "builtin sort": sort_from_scratch_counted(items),
        "topological": topological_heapsort(
            items,
            known_less,
            use_transitive_known_for_stats=use_transitive_known_for_stats,
        ),
        "lookahead": topological_heapsort_with_lookahead(
            items,
            known_less,
            use_transitive_known_for_stats=use_transitive_known_for_stats,
        ),
    }


def make_random_instance(
    n: int,
    m_known: int,
    *,
    seed: int | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Creates a random input permutation and m_known valid known comparisons.

    Does NOT build the O(n^2) list of all possible edges.
    """
    rng = random.Random(seed)

    items = list(range(1, n + 1))
    rng.shuffle(items)

    max_edges = n * (n - 1) // 2
    if m_known > max_edges:
        raise ValueError("m_known is larger than the number of possible comparisons.")

    known_set: set[tuple[int, int]] = set()

    while len(known_set) < m_known:
        a = rng.randint(1, n)
        b = rng.randint(1, n)

        if a == b:
            continue

        if a > b:
            a, b = b, a

        known_set.add((a, b))

    return items, list(known_set)


def make_known_chains_instance(
    n: int,
    num_chains: int,
    *,
    seed: int | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Creates an input where the known comparisons form num_chains sorted chains.

    If num_chains is small, topological heapsort should need about O(n log num_chains)
    comparisons instead of O(n log n).

    Example with num_chains=4:
        chain 0: 1, 5, 9, 13, ...
        chain 1: 2, 6, 10, 14, ...
        chain 2: 3, 7, 11, 15, ...
        chain 3: 4, 8, 12, 16, ...
    """
    import random

    rng = random.Random(seed)

    chains = [[] for _ in range(num_chains)]

    for x in range(1, n + 1):
        chains[(x - 1) % num_chains].append(x)

    known = []
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            known.append((a, b))

    items = list(range(1, n + 1))
    rng.shuffle(items)

    return items, known


# -----------------------------
# Example
# -----------------------------
# if __name__ == "__main__":
#     # items = [8, 3, 6, 1, 7, 2, 5, 4]

#     # # Already known comparisons: each tuple means a < b.
#     # known = [
#     #     (1, 3),
#     #     (2, 4),
#     #     (3, 6),
#     #     (4, 8),
#     #     (5, 7),
#     # ]

#     items = [9, 3, 7, 1, 8, 2, 6, 4, 5]

#     known = [
#         (1, 3),
#         (3, 6),
#         (6, 9),   # chain creates singleton-ish bottleneck structure

#         (2, 7),
#         (4, 8),
#     ]

#     items = list(range(1, 38))[::-1]

#     known = [(i, i + 1) for i in range(1, 37)]

#     results = compare_all(items, known)

#     for name, stats in results.items():
#         print(f"\n{name}")
#         # print("output:", stats.output)
#         print("total item comparisons:", stats.total_item_comparisons)
#         # print("additional item comparisons:", stats.additional_item_comparisons)
#         # print("known/implied comparisons used:", stats.known_or_implied_comparisons_used)
#         print("heap inserts:", stats.heap_inserts)
#         print("heap delete-mins:", stats.heap_delete_mins)
#         print("heap merges:", stats.heap_merges)


if __name__ == "__main__":
    # items, known = make_known_chains_instance(
    #     n=100_000,
    #     num_chains=10,
    #     seed=1,
    # )

    res = {"n log n": [], "builtin sort": [], "topological": [], "lookahead": []}

    for n in [100, 1000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]:  # , 100_000, 300_000, 500_000, 1_000_000]:

        items, known = make_random_instance(
            n=n,
            m_known=n,
            seed=123,
        )

        results = compare_all(items, known, use_transitive_known_for_stats=False)

        for name, stats in results.items():
            res[name].append((n, stats.total_item_comparisons))
            # print(f"\n{name}")
            # # print("output:", stats.output)
            # print("total item comparisons:", stats.total_item_comparisons)
            # # print("additional item comparisons:", stats.additional_item_comparisons)
            # # print("known/implied comparisons used:", stats.known_or_implied_comparisons_used)
            # print("heap inserts:", stats.heap_inserts)
            # print("heap delete-mins:", stats.heap_delete_mins)
            # print("heap merges:", stats.heap_merges)

        res["n log n"].append((n, n * int(log2(n))))

    import matplotlib.pyplot as plt

    for name, data in res.items():
        x, y = zip(*data)
        plt.plot(x, y, label=name)

    plt.xlabel("n")
    plt.ylabel("total item comparisons")
    plt.title("Total item comparisons vs n")
    plt.legend()
    plt.show()
