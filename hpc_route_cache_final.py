"""
Benchmark route caching for an HPC-style shortest-path workload.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Node = int
Weight = int
PathList = List[Node]
EdgeList = Dict[Node, List[Tuple[Node, Weight]]]


@dataclass(frozen=True)
class RouteResult:
    """Represents the result of one shortest-path lookup."""

    cost: float
    path: Tuple[Node, ...]
    from_cache: bool = False


class RouteGraph:
    """
    Weighted graph optimized for sparse connectivity.

    The graph is stored as an adjacency list. For sparse graphs, an adjacency
    list is usually more space-friendly than an adjacency matrix because it
    stores only real edges. The class also owns the route cache so the cache can
    be invalidated when the graph topology changes.
    """

    def __init__(self) -> None:
        self.edges: EdgeList = {}
        self.route_cache: Dict[Tuple[Node, Node, int], RouteResult] = {}
        self.graph_version = 0

    def add_link(self, left: Node, right: Node, weight: Weight) -> None:
        """Add an undirected weighted link and invalidate cached routes."""
        if weight < 0:
            raise ValueError("Dijkstra's algorithm requires non-negative weights.")

        self.edges.setdefault(left, []).append((right, weight))
        self.edges.setdefault(right, []).append((left, weight))
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        """
        Clear cached routes when topology changes.

        A cached path is only correct for the graph version that produced it.
        Clearing the cache is simple and safe. A production system might use a
        more selective invalidation policy, but this version keeps correctness
        easier to verify for a course project.
        """
        self.graph_version += 1
        self.route_cache.clear()

    def shortest_path_uncached(self, source: Node, destination: Node) -> RouteResult:
        """
        Baseline Dijkstra implementation.

        Every call performs a full shortest-path search. This is correct, but it
        can waste time when the same source and destination pair appears many
        times in a workload.
        """
        distances: Dict[Node, float] = {source: 0.0}
        previous: Dict[Node, Optional[Node]] = {source: None}
        priority_queue: List[Tuple[float, Node]] = [(0.0, source)]
        visited: set[Node] = set()

        while priority_queue:
            current_cost, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node == destination:
                break

            for neighbor, edge_weight in self.edges.get(current_node, []):
                candidate_cost = current_cost + edge_weight
                if candidate_cost < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate_cost
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (candidate_cost, neighbor))

        if destination not in distances:
            return RouteResult(cost=float("inf"), path=tuple())

        path: PathList = []
        cursor: Optional[Node] = destination
        while cursor is not None:
            path.append(cursor)
            cursor = previous.get(cursor)
        path.reverse()
        return RouteResult(cost=distances[destination], path=tuple(path))

    def shortest_path_cached(self, source: Node, destination: Node) -> RouteResult:
        """
        Optimized route lookup using a version-aware cache.

        The cache key includes graph_version. If links are added later, the
        graph_version changes and old routes cannot be reused accidentally.
        """
        cache_key = (source, destination, self.graph_version)
        cached_result = self.route_cache.get(cache_key)
        if cached_result is not None:
            return RouteResult(
                cost=cached_result.cost,
                path=cached_result.path,
                from_cache=True,
            )

        result = self.shortest_path_uncached(source, destination)
        self.route_cache[cache_key] = result
        return result


def build_sparse_graph(node_count: int, extra_edges: int, seed: int) -> RouteGraph:
    """Create a connected sparse graph with deterministic random edges."""
    rng = random.Random(seed)
    graph = RouteGraph()

    # A chain guarantees that every node is reachable.
    for node in range(node_count - 1):
        graph.add_link(node, node + 1, rng.randint(1, 20))

    # Random edges create alternate routes, similar to multiple possible paths.
    observed = {tuple(sorted((node, node + 1))) for node in range(node_count - 1)}
    added = 0
    while added < extra_edges:
        left = rng.randrange(node_count)
        right = rng.randrange(node_count)
        if left == right:
            continue

        edge_id = tuple(sorted((left, right)))
        if edge_id in observed:
            continue

        graph.add_link(left, right, rng.randint(1, 40))
        observed.add(edge_id)
        added += 1

    return graph


def build_repeated_workload(node_count: int, unique_pairs: int, repeats: int, seed: int) -> List[Tuple[Node, Node]]:
    """Build route requests that repeat enough to make caching measurable."""
    rng = random.Random(seed)
    pairs: List[Tuple[Node, Node]] = []

    while len(pairs) < unique_pairs:
        source = rng.randrange(node_count)
        destination = rng.randrange(node_count)
        if source != destination:
            pairs.append((source, destination))

    workload = pairs * repeats
    rng.shuffle(workload)
    return workload


def time_requests(graph: RouteGraph, workload: Iterable[Tuple[Node, Node]], use_cache: bool) -> Tuple[float, int]:
    """Measure total request time and count cache hits."""
    cache_hits = 0
    start = time.perf_counter()

    for source, destination in workload:
        if use_cache:
            result = graph.shortest_path_cached(source, destination)
            if result.from_cache:
                cache_hits += 1
        else:
            graph.shortest_path_uncached(source, destination)

    elapsed_seconds = time.perf_counter() - start
    return elapsed_seconds, cache_hits


def run_one_benchmark(nodes: int, extra_edges: int, unique_pairs: int, repeats: int, seed: int) -> Dict[str, float]:
    """Run one baseline-vs-optimized benchmark and return a metrics dictionary."""
    graph = build_sparse_graph(nodes, extra_edges, seed=seed)
    workload = build_repeated_workload(nodes, unique_pairs, repeats, seed=seed + 500)

    baseline_seconds, _ = time_requests(graph, workload, use_cache=False)
    cached_seconds, cache_hits = time_requests(graph, workload, use_cache=True)

    baseline_ms = baseline_seconds * 1000
    cached_ms = cached_seconds * 1000
    improvement = ((baseline_ms - cached_ms) / baseline_ms) * 100 if baseline_ms else 0.0

    return {
        "nodes": nodes,
        "edges": (nodes - 1) + extra_edges,
        "requests": len(workload),
        "baseline_ms": baseline_ms,
        "cached_ms": cached_ms,
        "improvement_percent": improvement,
        "cache_hits": cache_hits,
    }


def print_results(results: List[Dict[str, float]]) -> None:
    """Print benchmark results in a readable table."""
    print("HPC Route Cache Optimization Benchmark")
    print("Baseline: Dijkstra runs for every request")
    print("Optimized: repeated route requests use a graph-versioned cache")
    print()
    print(f"{'Nodes':>8} {'Edges':>8} {'Requests':>10} {'Baseline ms':>14} {'Cached ms':>12} {'Improve %':>10} {'Cache hits':>11}")
    print("-" * 84)
    for item in results:
        print(
            f"{int(item['nodes']):8d} "
            f"{int(item['edges']):8d} "
            f"{int(item['requests']):10d} "
            f"{item['baseline_ms']:14.3f} "
            f"{item['cached_ms']:12.3f} "
            f"{item['improvement_percent']:10.1f} "
            f"{int(item['cache_hits']):11d}"
        )


def save_csv(results: List[Dict[str, float]], path: str) -> None:
    """Save benchmark metrics to a CSV file for the report appendix."""
    output_path = Path(path)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV results to: {output_path.resolve()}")


def run_demo() -> None:
    """Show one cached route so the optimization can be explained during presentation."""
    graph = build_sparse_graph(node_count=20, extra_edges=35, seed=7)
    first = graph.shortest_path_cached(0, 15)
    second = graph.shortest_path_cached(0, 15)

    print("Demo route request: node 0 to node 15")
    print(f"First request cost: {first.cost}, path: {list(first.path)}, from_cache={first.from_cache}")
    print(f"Second request cost: {second.cost}, path: {list(second.path)}, from_cache={second.from_cache}")
    print("The second request is served from cache because the graph did not change.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark route caching for an HPC-style shortest-path workload.")
    parser.add_argument("--nodes", type=int, default=None, help="Run one custom graph size.")
    parser.add_argument("--extra-edges", type=int, default=None, help="Number of additional random edges for custom mode.")
    parser.add_argument("--unique-pairs", type=int, default=None, help="Number of unique route pairs for custom mode.")
    parser.add_argument("--repeats", type=int, default=None, help="How many times to repeat each unique pair.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed used for deterministic graph/workload generation.")
    parser.add_argument("--csv", type=str, default="", help="Optional path to save benchmark results as CSV.")
    parser.add_argument("--demo", action="store_true", help="Print a simple route lookup demo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        run_demo()
        return

    if args.nodes is not None:
        if args.extra_edges is None or args.unique_pairs is None or args.repeats is None:
            raise SystemExit("Custom mode requires --extra-edges, --unique-pairs, and --repeats.")
        scenarios = [(args.nodes, args.extra_edges, args.unique_pairs, args.repeats, args.seed)]
    else:
        scenarios = [
            (150, 450, 75, 8, 101),
            (600, 1800, 150, 8, 102),
            (1200, 3600, 200, 8, 103),
        ]

    results = [run_one_benchmark(*scenario) for scenario in scenarios]
    print_results(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
