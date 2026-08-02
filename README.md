# MSCS532_Project1
Final Project Part 1: Optimization Technique and Implementation Project Report

HPC Optimization
================
Purpose
-------
This program demonstrates one practical data structure optimization in a small
HPC-style workload. The baseline version runs Dijkstra's shortest-path algorithm
for every route request. The optimized version stores previous route results in a
cache and reuses them when the same route is requested again and the graph has
not changed.

Why this matches the assignment
-------------------------------
- Data structure optimization: adjacency list + dictionary cache + heap priority queue.
- HPC connection: avoids repeated expensive graph computation in repeated workloads.
- Implementation comparison: baseline runtime vs optimized runtime.
- Correctness concern: graph-version invalidation prevents stale cached routes.

Commands to run:

```python
1. Run the default benchmark:
   python hpc_route_cache_final.py

2. Run a small demo and print one route:
   python hpc_route_cache_final.py --demo

3. Run a custom benchmark:
   python hpc_route_cache_final.py --nodes 800 --extra-edges 2400 --unique-pairs 160 --repeats 8

4. Save benchmark results to CSV:
   python hpc_route_cache_final.py --csv hpc_benchmark_results.csv
```
