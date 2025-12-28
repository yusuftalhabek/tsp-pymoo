from __future__ import annotations
import argparse
from datetime import datetime
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Configure cache paths before importing matplotlib to avoid permission issues.
MPL_DIR = Path(__file__).with_name(".matplotlib")
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_DIR))
os.environ.setdefault("FONTCONFIG_PATH", str(MPL_DIR))

import matplotlib

# Use non-interactive backend for headless execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.termination import get_termination


@dataclass
class TSPData:
    coords: np.ndarray  # shape (n, 2)
    dist_matrix: np.ndarray  # shape (n, n)


def load_data(city_path: Path, distance_path: Path) -> TSPData:
    #Load coordinates and distance matrix from the provided text files.
    city_raw = np.loadtxt(city_path, dtype=float)
    coords = city_raw[:, 1:3]  # drop city id column
    dist_matrix = np.loadtxt(distance_path, dtype=float)
    return TSPData(coords=coords, dist_matrix=dist_matrix)


def route_length(route: List[int], dist_matrix: np.ndarray) -> float:
    #Compute total length of a round trip route.
    return float(sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)))


class FixedStartTSP(ElementwiseProblem):
    #TSP variant that fixes the starting city and returns there.

    def __init__(self, dist_matrix: np.ndarray, start_idx: int):
        self.dist_matrix = dist_matrix
        self.start_idx = start_idx
        n_cities = dist_matrix.shape[0]
        self.remaining = [i for i in range(n_cities) if i != start_idx]
        super().__init__(
            n_var=len(self.remaining),
            n_obj=1,
            xl=0,
            xu=len(self.remaining) - 1,
            type_var=int,
            permutation=True,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        visit_order = [self.remaining[i] for i in x]
        route = [self.start_idx] + visit_order + [self.start_idx]
        out["F"] = route_length(route, self.dist_matrix)



def make_ga(pop_size: int) -> GA:
    #Configure a permutation-aware GA.
    return GA(
        pop_size=pop_size,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(prob=0.9),
        mutation=InversionMutation(prob=0.2),
        eliminate_duplicates=True,
    )


def make_nsga2(pop_size: int) -> NSGA2:
    #Configure NSGA-II for single-objective TSP to contrast GA behavior.
    return NSGA2(
        pop_size=pop_size,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(prob=0.9),
        mutation=InversionMutation(prob=0.3),
        eliminate_duplicates=True,
    )

def decode_route(solution: Iterable[int], problem: FixedStartTSP) -> List[int]:
    #Translate a permutation solution back to city indices (0-based) including start/end.
    seq = np.asarray(solution, dtype=int).ravel()
    visit_order = [problem.remaining[int(i)] for i in seq]
    return [problem.start_idx] + visit_order + [problem.start_idx]


def plot_route(
    route: List[int],
    coords: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    #Plot a TSP route on a 2D plane and save to disk.
    points = coords[route]
    plt.figure(figsize=(7, 6))
    plt.plot(points[:, 0], points[:, 1], "-o", color="tab:blue", alpha=0.8)

    # Highlight start/end city.
    plt.scatter(points[0, 0], points[0, 1], color="red", marker="s", s=100, label="Start/End")
    for idx, city in enumerate(route[:-1]):
        plt.text(points[idx, 0] + 1.2, points[idx, 1] + 1.2, str(city + 1), fontsize=8)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def solve_for_start(
    tsp: TSPData,
    start_city: int,
    n_generations: int,
    pop_size: int,
    seed: int,
    run_dir: Path,
) -> Dict[str, Dict[str, object]]:
    #Run both algorithms for a single starting city and return their results.
    start_idx = start_city - 1  # convert to 0-based
    problem = FixedStartTSP(tsp.dist_matrix, start_idx=start_idx)
    termination = get_termination("n_gen", n_generations)

    runs = {
        "ga": make_ga(pop_size),
        "nsga2": make_nsga2(pop_size),
    }

    results: Dict[str, Dict[str, object]] = {}

    for name, algo in runs.items():
        res = minimize(
            problem,
            algo,
            termination=termination,
            seed=seed,
            verbose=False,
        )
        route = decode_route(res.X, problem)
        # res.F can be nested; pull the first scalar safely.
        length = float(np.atleast_1d(res.F).ravel()[0])

        title = f"{name.upper()} | start city {start_city} | length {length:.2f}"
        plot_path = run_dir / f"route_start{start_city}_{name}.png"
        plot_route(route, tsp.coords, title, plot_path)

        results[name] = {
            "route": route,
            "length": length,
            "plot": plot_path,
            "pymoo_result": res,
        }

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve TSP instance using PyMOO.")
    parser.add_argument(
        "--starts",
        type=str,
        default="1,5,10,20,30",
        help="Comma-separated list of starting city IDs (1-based). Must include at least 5.",
    )
    parser.add_argument("--generations", type=int, default=250, help="Generations per algorithm run.")
    parser.add_argument("--pop-size", type=int, default=120, help="Population size for both algorithms.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory to store route plots and any outputs.",
    )
    parser.add_argument(
        "--city-file",
        type=Path,
        default=Path("cityData.txt"),
        help="Path to the city coordinate file.",
    )
    parser.add_argument(
        "--distance-file",
        type=Path,
        default=Path("intercityDistance.txt"),
        help="Path to the distance matrix file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_cities = [int(x) for x in args.starts.split(",") if x.strip()]
    if len(start_cities) < 5:
        raise ValueError("Please provide at least five starting cities via --starts.")

    tsp = load_data(args.city_file, args.distance_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.out_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Tuple[int, str, float, List[int], Path]] = []

    for start_city in start_cities:
        results = solve_for_start(
            tsp=tsp,
            start_city=start_city,
            n_generations=args.generations,
            pop_size=args.pop_size,
            seed=args.seed,
            run_dir=run_dir,
        )
        # pick the better route between the two algorithms
        best_name, best = min(results.items(), key=lambda item: item[1]["length"])
        summary.append((start_city, best_name, best["length"], best["route"], best["plot"]))

        print(f"Start city {start_city}:")
        for name, res in results.items():
            route_ids = [c + 1 for c in res["route"]]  # convert to 1-based for readability
            print(f"  {name:<6} length={res['length']:.2f} route={route_ids} plot={res['plot']}")

    print("\nBest routes per start city (using the better of GA / NSGA-II):")
    for start_city, algo_name, length, route, plot_path in summary:
        route_ids = [c + 1 for c in route]
        print(f"  start {start_city}: {algo_name.upper()} length={length:.2f} route={route_ids} plot={plot_path}")
    print(f"\nPlots saved under: {run_dir}")


if __name__ == "__main__":
    main()

