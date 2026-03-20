import argparse
import json
import math
import pickle as pkl
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataset import HandleRSSDataset, NCDataset, fix_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_DIR = PROJECT_ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic graphs for scalability experiments and export them in BucketGNN format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=["node_scale", "density_scale"],
        required=True,
        help="node_scale: vary node count while fixing mean/max degree; density_scale: vary density while fixing node count.",
    )
    parser.add_argument(
        "--generator",
        choices=["er", "kronecker"],
        default="kronecker",
        help="Synthetic graph generator.",
    )
    parser.add_argument(
        "--num_nodes_list",
        type=int,
        nargs="+",
        default=None,
        help="Node counts for node_scale experiments.",
    )
    parser.add_argument(
        "--fixed_num_nodes",
        type=int,
        default=None,
        help="Fixed node count for density_scale experiments.",
    )
    parser.add_argument(
        "--mean_degree",
        type=float,
        default=None,
        help="Fixed mean degree for node_scale experiments.",
    )
    parser.add_argument(
        "--mean_degree_list",
        type=float,
        nargs="+",
        default=None,
        help="Mean degree list for density_scale experiments.",
    )
    parser.add_argument(
        "--density_list",
        type=float,
        nargs="+",
        default=None,
        help="Optional density list for density_scale experiments. If provided, mean_degree_list is ignored.",
    )
    parser.add_argument(
        "--max_degree",
        type=int,
        default=None,
        help="Degree cap used to keep the maximum degree controlled.",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=128,
        help="Feature dimension of synthetic node features.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=7,
        help="Number of synthetic classes.",
    )
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0.35,
        help="Noise strength for prototype-based features.",
    )
    parser.add_argument(
        "--train_prop",
        type=float,
        default=0.2,
        help="Training split proportion.",
    )
    parser.add_argument(
        "--valid_prop",
        type=float,
        default=0.2,
        help="Validation split proportion.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many replicas to generate per configuration.",
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default="synthetic",
        help="Prefix of the exported dataset directory name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_SAVE_DIR),
        help="Output directory for generated BucketGNN datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite metadata and regenerate adjacency-derived files when the dataset folder already exists.",
    )
    parser.add_argument(
        "--initiator",
        type=float,
        nargs=4,
        default=[0.57, 0.19, 0.19, 0.05],
        metavar=("A", "B", "C", "D"),
        help="Kronecker/R-MAT initiator matrix in row-major order.",
    )
    return parser.parse_args()


def ensure_valid_args(args):
    if args.train_prop <= 0 or args.valid_prop <= 0 or args.train_prop + args.valid_prop >= 1:
        raise ValueError("train_prop and valid_prop must be > 0, and train_prop + valid_prop must be < 1.")
    if args.fixed_num_nodes is not None and args.fixed_num_nodes <= 0:
        raise ValueError("fixed_num_nodes must be positive.")
    if args.num_nodes_list is not None and any(num_nodes <= 0 for num_nodes in args.num_nodes_list):
        raise ValueError("All values in num_nodes_list must be positive.")
    if args.num_features <= 0:
        raise ValueError("num_features must be positive.")
    if args.num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if args.max_degree is not None and args.max_degree <= 0:
        raise ValueError("max_degree must be positive.")
    if args.mean_degree is not None and args.mean_degree < 0:
        raise ValueError("mean_degree must be non-negative.")
    if args.mean_degree_list is not None and any(mean_degree < 0 for mean_degree in args.mean_degree_list):
        raise ValueError("All values in mean_degree_list must be non-negative.")
    if args.density_list is not None and any(density < 0 or density > 1 for density in args.density_list):
        raise ValueError("Each density in density_list must be between 0 and 1.")

    if args.experiment == "node_scale":
        if not args.num_nodes_list:
            raise ValueError("node_scale requires --num_nodes_list.")
        if args.mean_degree is None:
            raise ValueError("node_scale requires --mean_degree.")
        if args.max_degree is None:
            raise ValueError("node_scale requires --max_degree.")
    else:
        if args.fixed_num_nodes is None:
            raise ValueError("density_scale requires --fixed_num_nodes.")
        if args.mean_degree_list is None and args.density_list is None:
            raise ValueError("density_scale requires --mean_degree_list or --density_list.")


def balanced_labels(num_nodes: int, num_classes: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.arange(num_nodes, dtype=np.int64) % num_classes
    rng.shuffle(labels)
    return labels


def prototype_features(
    labels: np.ndarray,
    num_features: int,
    num_classes: int,
    feature_noise: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    centers = rng.normal(loc=0.0, scale=1.0, size=(num_classes, num_features)).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=feature_noise, size=(labels.shape[0], num_features)).astype(np.float32)
    features = centers[labels] + noise
    features = torch.tensor(features, dtype=torch.float32)
    return F.normalize(features, p=2, dim=1)


def build_random_split(
    num_nodes: int,
    train_prop: float,
    valid_prop: float,
    rng: np.random.Generator,
):
    perm = rng.permutation(num_nodes)
    train_end = int(num_nodes * train_prop)
    valid_end = train_end + int(num_nodes * valid_prop)
    split_idx = {
        "train": torch.tensor(np.sort(perm[:train_end]), dtype=torch.long),
        "valid": torch.tensor(np.sort(perm[train_end:valid_end]), dtype=torch.long),
        "test": torch.tensor(np.sort(perm[valid_end:]), dtype=torch.long),
    }
    return split_idx


def target_num_edges(num_nodes: int, mean_degree: float) -> int:
    max_possible = num_nodes * (num_nodes - 1) // 2
    target = int(round(num_nodes * mean_degree / 2.0))
    return max(0, min(target, max_possible))


def sample_unique_pairs(
    num_nodes: int,
    target_edges_count: int,
    rng: np.random.Generator,
) -> Set[Tuple[int, int]]:
    max_possible = num_nodes * (num_nodes - 1) // 2
    if target_edges_count > max_possible:
        raise ValueError("Requested number of edges exceeds the number of possible undirected edges.")

    edges: Set[Tuple[int, int]] = set()
    if target_edges_count == 0:
        return edges

    while len(edges) < target_edges_count:
        remaining = target_edges_count - len(edges)
        batch_size = max(remaining * 3, 1024)
        src = rng.integers(0, num_nodes, size=batch_size, endpoint=False)
        dst = rng.integers(0, num_nodes, size=batch_size, endpoint=False)
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        if src.size == 0:
            continue
        lo = np.minimum(src, dst)
        hi = np.maximum(src, dst)
        for u, v in zip(lo.tolist(), hi.tolist()):
            edges.add((u, v))
            if len(edges) >= target_edges_count:
                break
    return edges


def sample_erdos_renyi_edges(
    num_nodes: int,
    mean_degree: float,
    rng: np.random.Generator,
) -> Set[Tuple[int, int]]:
    return sample_unique_pairs(num_nodes, target_num_edges(num_nodes, mean_degree), rng)


def sample_kronecker_edges(
    num_nodes: int,
    mean_degree: float,
    initiator: Sequence[float],
    rng: np.random.Generator,
) -> Set[Tuple[int, int]]:
    probs = np.array(initiator, dtype=np.float64).reshape(2, 2)
    if np.any(probs < 0):
        raise ValueError("Kronecker initiator entries must be non-negative.")
    total = probs.sum()
    if total <= 0:
        raise ValueError("Kronecker initiator sum must be positive.")
    probs = (probs / total).reshape(-1)

    scale = int(math.ceil(math.log2(max(num_nodes, 2))))
    matrix_size = 1 << scale
    needed_edges = target_num_edges(num_nodes, mean_degree)

    edges: Set[Tuple[int, int]] = set()
    if needed_edges == 0:
        return edges

    while len(edges) < needed_edges:
        remaining = needed_edges - len(edges)
        batch_size = max(remaining * 4, 1024)
        rows = np.zeros(batch_size, dtype=np.int64)
        cols = np.zeros(batch_size, dtype=np.int64)
        step = matrix_size // 2

        for _ in range(scale):
            quadrant = rng.choice(4, size=batch_size, p=probs)
            rows += ((quadrant == 2) | (quadrant == 3)) * step
            cols += ((quadrant == 1) | (quadrant == 3)) * step
            step //= 2

        mask = (rows < num_nodes) & (cols < num_nodes) & (rows != cols)
        rows = rows[mask]
        cols = cols[mask]
        if rows.size == 0:
            continue

        lo = np.minimum(rows, cols)
        hi = np.maximum(rows, cols)
        for u, v in zip(lo.tolist(), hi.tolist()):
            edges.add((u, v))
            if len(edges) >= needed_edges:
                break

    return edges


def build_adj_sets(num_nodes: int, edges: Iterable[Tuple[int, int]]) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def edge_list_from_adj(adj: Sequence[Set[int]]) -> List[Tuple[int, int]]:
    edges = []
    for u, neighbors in enumerate(adj):
        for v in neighbors:
            if u < v:
                edges.append((u, v))
    return edges


def trim_to_max_degree(adj: List[Set[int]], max_degree: int, rng: np.random.Generator):
    if max_degree is None:
        return

    overloaded = [idx for idx, neighbors in enumerate(adj) if len(neighbors) > max_degree]
    while overloaded:
        node = overloaded.pop()
        while len(adj[node]) > max_degree:
            neighbor = int(rng.choice(list(adj[node])))
            adj[node].remove(neighbor)
            adj[neighbor].remove(node)
            if len(adj[neighbor]) > max_degree and neighbor not in overloaded:
                overloaded.append(neighbor)


def adjust_to_target_mean_degree(
    adj: List[Set[int]],
    target_mean_degree_value: float,
    max_degree: int,
    rng: np.random.Generator,
):
    num_nodes = len(adj)
    target_edges_count = target_num_edges(num_nodes, target_mean_degree_value)
    current_edges = len(edge_list_from_adj(adj))

    if current_edges > target_edges_count:
        edges = edge_list_from_adj(adj)
        rng.shuffle(edges)
        for u, v in edges:
            if current_edges <= target_edges_count:
                break
            if v in adj[u]:
                adj[u].remove(v)
                adj[v].remove(u)
                current_edges -= 1

    attempts = 0
    max_attempts = max(target_edges_count * 20, 10000)
    degree_cap = max_degree if max_degree is not None else num_nodes - 1

    while current_edges < target_edges_count and attempts < max_attempts:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        attempts += 1
        if u == v or v in adj[u]:
            continue
        if len(adj[u]) >= degree_cap or len(adj[v]) >= degree_cap:
            continue
        adj[u].add(v)
        adj[v].add(u)
        current_edges += 1

    if current_edges < target_edges_count:
        print(
            f"[Warning] Could not reach the exact target mean degree after degree-capping. "
            f"target_edges={target_edges_count}, current_edges={current_edges}."
        )


def adjacency_to_edge_index(adj: Sequence[Set[int]]) -> torch.Tensor:
    rows = []
    cols = []
    for u, neighbors in enumerate(adj):
        for v in sorted(neighbors):
            rows.append(u)
            cols.append(v)
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def graph_stats(adj: Sequence[Set[int]]):
    degrees = np.array([len(neighbors) for neighbors in adj], dtype=np.int64)
    num_nodes = len(adj)
    num_edges = int(degrees.sum() // 2)
    density = 0.0 if num_nodes <= 1 else (2.0 * num_edges) / (num_nodes * (num_nodes - 1))
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "mean_degree": float(degrees.mean()) if num_nodes > 0 else 0.0,
        "max_degree": int(degrees.max()) if num_nodes > 0 else 0,
        "min_degree": int(degrees.min()) if num_nodes > 0 else 0,
        "density": density,
    }


def generate_topology(
    generator: str,
    num_nodes: int,
    mean_degree: float,
    max_degree: int,
    initiator: Sequence[float],
    rng: np.random.Generator,
) -> List[Set[int]]:
    if max_degree is not None and mean_degree > max_degree:
        raise ValueError("mean_degree cannot be larger than max_degree when degree capping is enabled.")

    if generator == "er":
        edges = sample_erdos_renyi_edges(num_nodes, mean_degree, rng)
    elif generator == "kronecker":
        edges = sample_kronecker_edges(num_nodes, mean_degree, initiator, rng)
    else:
        raise ValueError(f"Unsupported generator: {generator}")

    adj = build_adj_sets(num_nodes, edges)
    trim_to_max_degree(adj, max_degree=max_degree, rng=rng)
    adjust_to_target_mean_degree(adj, target_mean_degree_value=mean_degree, max_degree=max_degree, rng=rng)
    trim_to_max_degree(adj, max_degree=max_degree, rng=rng)
    return adj


def build_dataset(
    dataset_name: str,
    num_nodes: int,
    mean_degree: float,
    max_degree: int,
    num_features: int,
    num_classes: int,
    feature_noise: float,
    generator: str,
    initiator: Sequence[float],
    train_prop: float,
    valid_prop: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    labels = balanced_labels(num_nodes, num_classes, rng)
    node_feat = prototype_features(labels, num_features, num_classes, feature_noise, rng)
    adj = generate_topology(generator, num_nodes, mean_degree, max_degree, initiator, rng)
    split_idx = build_random_split(num_nodes, train_prop, valid_prop, rng)

    dataset = NCDataset(dataset_name)
    dataset.graph = {
        "edge_index": adjacency_to_edge_index(adj),
        "node_feat": node_feat,
        "edge_feat": None,
        "num_nodes": num_nodes,
    }
    dataset.label = torch.tensor(labels, dtype=torch.long).unsqueeze(1)
    return dataset, split_idx, adj, graph_stats(adj)


class SyntheticRSSDataset(HandleRSSDataset):
    def __init__(self, *args, synthetic_adj: Sequence[Set[int]], **kwargs):
        super().__init__(*args, **kwargs)
        self.synthetic_adj = [sorted(list(neighbors)) for neighbors in synthetic_adj]

    def _build_adjacency(self):
        path_adj_list = Path(self.data_path) / "edge_info_plaintext" / "adj_list.pkl"
        path_edge_weight = Path(self.data_path) / "edge_info_plaintext" / "edge_weight.pkl"
        path_adj_list.parent.mkdir(parents=True, exist_ok=True)

        if path_adj_list.exists() and path_edge_weight.exists():
            with path_adj_list.open("rb") as f:
                try:
                    adj_list = pkl.load(f)
                except Exception:
                    f.seek(0)
                    adj_list = json.load(f)
            with path_edge_weight.open("rb") as f:
                try:
                    edge_weight = pkl.load(f)
                except Exception:
                    f.seek(0)
                    edge_weight = json.load(f)
            print(f"Loaded adjacency list from {path_adj_list}")
            return adj_list, edge_weight

        adj_list = [list(neighbors) for neighbors in self.synthetic_adj]
        edge_weight = [[1] * len(neighbors) for neighbors in adj_list]

        with path_adj_list.open("wb") as f:
            pkl.dump(adj_list, f)
        with path_edge_weight.open("wb") as f:
            pkl.dump(edge_weight, f)
        return adj_list, edge_weight


def dataset_name_from_config(
    prefix: str,
    generator: str,
    experiment: str,
    num_nodes: int,
    mean_degree: float,
    max_degree: int,
    repeat_idx: int,
):
    mean_degree_tag = str(mean_degree).replace(".", "p")
    max_degree_tag = "none" if max_degree is None else str(max_degree)
    return (
        f"{prefix}_{generator}_{experiment}_"
        f"n{num_nodes}_md{mean_degree_tag}_cap{max_degree_tag}_r{repeat_idx}"
    )


def save_metadata(dataset_dir: Path, config: dict, stats: dict):
    metadata_path = dataset_dir / "synthetic_config.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump({"config": config, "stats": stats}, f, indent=2)


def run_one_configuration(args, num_nodes: int, mean_degree: float, repeat_idx: int):
    seed = args.seed + repeat_idx
    dataset_name = dataset_name_from_config(
        prefix=args.name_prefix,
        generator=args.generator,
        experiment=args.experiment,
        num_nodes=num_nodes,
        mean_degree=mean_degree,
        max_degree=args.max_degree,
        repeat_idx=repeat_idx,
    )

    print(f"\n[Generate] {dataset_name}")
    dataset, split_idx, adj, stats = build_dataset(
        dataset_name=dataset_name,
        num_nodes=num_nodes,
        mean_degree=mean_degree,
        max_degree=args.max_degree,
        num_features=args.num_features,
        num_classes=args.num_classes,
        feature_noise=args.feature_noise,
        generator=args.generator,
        initiator=args.initiator,
        train_prop=args.train_prop,
        valid_prop=args.valid_prop,
        seed=seed,
    )
    print(f"[Stats] {stats}")

    handler = SyntheticRSSDataset(
        dataset_name=dataset_name,
        save_dir=str(Path(args.output_dir).resolve()),
        dataset=dataset,
        split_idx=split_idx,
        max_degree=stats["max_degree"],
        mean_degree=stats["mean_degree"],
        synthetic_adj=adj,
    )
    handler.save_train_data()
    handler.save_test_data(clean_adj_list_files=args.overwrite)

    dataset_dir = Path(args.output_dir).resolve() / dataset_name
    save_metadata(
        dataset_dir,
        config={
            "experiment": args.experiment,
            "generator": args.generator,
            "num_nodes": num_nodes,
            "target_mean_degree": mean_degree,
            "max_degree": args.max_degree,
            "num_features": args.num_features,
            "num_classes": args.num_classes,
            "feature_noise": args.feature_noise,
            "train_prop": args.train_prop,
            "valid_prop": args.valid_prop,
            "seed": seed,
            "initiator": args.initiator,
        },
        stats=stats,
    )


def build_experiment_grid(args):
    if args.experiment == "node_scale":
        return [(num_nodes, float(args.mean_degree)) for num_nodes in args.num_nodes_list]

    if args.density_list is not None:
        return [
            (args.fixed_num_nodes, float(density) * max(args.fixed_num_nodes - 1, 0))
            for density in args.density_list
        ]
    return [(args.fixed_num_nodes, float(mean_degree)) for mean_degree in args.mean_degree_list]


def main():
    args = parse_args()
    ensure_valid_args(args)
    fix_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_grid = build_experiment_grid(args)
    print(f"Output directory: {output_dir}")
    print(f"Experiment grid: {experiment_grid}")

    for num_nodes, mean_degree in experiment_grid:
        for repeat_idx in range(args.repeats):
            run_one_configuration(args, num_nodes=num_nodes, mean_degree=mean_degree, repeat_idx=repeat_idx)


if __name__ == "__main__":
    main()