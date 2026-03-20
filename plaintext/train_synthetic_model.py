import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from model import resolve_device, write_model_file
from synthetic_dataset import (
    SyntheticRSSDataset,
    build_dataset,
    build_experiment_grid,
    dataset_name_from_config,
    ensure_valid_args,
    save_metadata,
)


class Net(torch.nn.Module):
    def __init__(self, feature_dim, label_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.gcn1 = GCNConv(feature_dim, hidden_dim, bias=True)
        self.gcn2 = GCNConv(hidden_dim, label_dim, bias=True)
        self.dropout = dropout
        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.gcn1.lin.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.gcn2.lin.weight, nonlinearity="relu")
        if self.gcn1.bias is not None:
            nn.init.zeros_(self.gcn1.bias)
        if self.gcn2.bias is not None:
            nn.init.zeros_(self.gcn2.bias)

    def forward(self, dataset):
        x = dataset.graph["node_feat"]
        edge_index = dataset.graph["edge_index"]
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        return F.log_softmax(x, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GCN on synthetic graphs for scalability experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--experiment", choices=["node_scale", "density_scale"], required=True)
    parser.add_argument("--generator", choices=["er", "kronecker"], default="kronecker")
    parser.add_argument("--num_nodes_list", type=int, nargs="+", default=None)
    parser.add_argument("--fixed_num_nodes", type=int, default=None)
    parser.add_argument("--mean_degree", type=float, default=None)
    parser.add_argument("--mean_degree_list", type=float, nargs="+", default=None)
    parser.add_argument("--density_list", type=float, nargs="+", default=None)
    parser.add_argument("--max_degree", type=int, default=None)
    parser.add_argument("--initiator", type=float, nargs=4, default=[0.57, 0.19, 0.19, 0.05])

    parser.add_argument("--num_features", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--feature_noise", type=float, default=0.35)
    parser.add_argument("--train_prop", type=float, default=0.2)
    parser.add_argument("--valid_prop", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=1, help="Graph replicas per configuration.")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--run", type=int, default=5, help="Training runs per synthetic graph.")
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--name_prefix", type=str, default="synthetic")
    parser.add_argument("--save_model", action="store_true", help="Save MPC-compatible A/B/C parameter files.")
    parser.add_argument("--save_results", action="store_true", help="Save json summary under output_dir/synthetic_results/.")
    parser.add_argument(
        "--save_dataset",
        action="store_true",
        help="Export synthetic datasets to BucketGNN format (dataset_info.txt, train/test shares, adjacency, node_state).",
    )

    return parser.parse_args()


def set_run_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_run(dataset, split_idx, args, run_seed: int, run_idx: int):
    set_run_seed(run_seed)
    device = resolve_device(args.device)

    dataset.graph["node_feat"] = dataset.graph["node_feat"].to(device)
    dataset.graph["edge_index"] = dataset.graph["edge_index"].to(device)
    dataset.label = dataset.label.to(device)

    train_idx = split_idx["train"].to(device)
    valid_idx = torch.cat([split_idx["train"], split_idx["valid"]], dim=-1).to(device)
    test_idx = split_idx["test"].to(device)

    feature_dim = dataset.graph["node_feat"].shape[1]
    label_dim = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    model = Net(feature_dim, label_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    true_label = dataset.label.squeeze(1).long()
    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset)
        loss = nn.NLLLoss()(out[train_idx], true_label[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(dataset)
            pred = logits.argmax(dim=1)

            val_correct = pred[valid_idx].eq(true_label[valid_idx]).sum().item()
            val_acc = val_correct / valid_idx.shape[0]

            test_correct = pred[test_idx].eq(true_label[test_idx]).sum().item()
            test_acc = test_correct / test_idx.shape[0]

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if (epoch + 1) % max(args.epochs // 10, 1) == 0 or epoch == 0:
            print(
                f"Run {run_idx + 1}/{args.run} Epoch {epoch + 1}/{args.epochs} "
                f"loss={loss.item():.4f} val={val_acc:.4f} test={test_acc:.4f}"
            )

    return model, float(best_val_acc), float(best_test_acc)


def save_model_shares(model: Net, output_root: Path, dataset_name: str, run_idx: int):
    model_dir = output_root / dataset_name / "network" / "gcn" / f"run_{run_idx}"
    model_dir.mkdir(parents=True, exist_ok=True)

    for existing in model_dir.iterdir():
        if existing.is_file():
            existing.unlink()

    parameters = {
        "fc_weight_1": model.gcn1.lin.weight.data.cpu().numpy(),
        "fc_bias_1": model.gcn1.bias.data.cpu().numpy(),
        "fc_weight_2": model.gcn2.lin.weight.data.cpu().numpy(),
        "fc_bias_2": model.gcn2.bias.data.cpu().numpy(),
    }

    for name, param in parameters.items():
        param_data = param.tolist()
        zero_data = np.zeros_like(param).tolist()
        write_model_file(str(model_dir / f"{name}_A"), param_data)
        write_model_file(str(model_dir / f"{name}_B"), zero_data)
        write_model_file(str(model_dir / f"{name}_C"), zero_data)


def maybe_write_results(output_root: Path, result_rows):
    if not result_rows:
        return
    results_dir = output_root / "synthetic_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "gcn_scalability_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result_rows, f, indent=2)
    print(f"Saved summary to: {output_file}")


def export_synthetic_dataset(
    output_root: Path,
    dataset_name: str,
    dataset,
    split_idx,
    adj,
    stats,
    args,
    graph_seed: int,
    mean_degree: float,
):
    handler = SyntheticRSSDataset(
        dataset_name=dataset_name,
        save_dir=str(output_root),
        dataset=dataset,
        split_idx=split_idx,
        max_degree=stats["max_degree"],
        mean_degree=stats["mean_degree"],
        synthetic_adj=adj,
    )
    handler.save_train_data()
    handler.save_test_data(clean_adj_list_files=False)

    save_metadata(
        output_root / dataset_name,
        config={
            "experiment": args.experiment,
            "generator": args.generator,
            "num_nodes": dataset.graph["num_nodes"],
            "target_mean_degree": mean_degree,
            "max_degree": args.max_degree,
            "num_features": args.num_features,
            "num_classes": args.num_classes,
            "feature_noise": args.feature_noise,
            "train_prop": args.train_prop,
            "valid_prop": args.valid_prop,
            "seed": graph_seed,
            "initiator": args.initiator,
        },
        stats=stats,
    )


def main():
    args = parse_args()
    ensure_valid_args(args)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_grid = build_experiment_grid(args)
    result_rows = []

    print(f"Output directory: {output_root}")
    print(f"Experiment grid: {experiment_grid}")

    for num_nodes, mean_degree in experiment_grid:
        for repeat_idx in range(args.repeats):
            graph_seed = args.seed + repeat_idx
            dataset_name = dataset_name_from_config(
                prefix=args.name_prefix,
                generator=args.generator,
                experiment=args.experiment,
                num_nodes=num_nodes,
                mean_degree=mean_degree,
                max_degree=args.max_degree,
                repeat_idx=repeat_idx,
            )

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
                seed=graph_seed,
            )

            if args.save_dataset:
                export_synthetic_dataset(
                    output_root=output_root,
                    dataset_name=dataset_name,
                    dataset=dataset,
                    split_idx=split_idx,
                    adj=adj,
                    stats=stats,
                    args=args,
                    graph_seed=graph_seed,
                    mean_degree=mean_degree,
                )

            run_best_vals = []
            run_best_tests = []
            print(f"\n=== Training on {dataset_name} ===")

            for run_idx in range(args.run):
                run_seed = graph_seed + 10000 + run_idx
                model, best_val_acc, best_test_acc = train_one_run(
                    dataset=dataset,
                    split_idx=split_idx,
                    args=args,
                    run_seed=run_seed,
                    run_idx=run_idx,
                )
                run_best_vals.append(best_val_acc)
                run_best_tests.append(best_test_acc)
                print(
                    f"Run {run_idx + 1}/{args.run} best_val={best_val_acc:.4f} "
                    f"best_test@best_val={best_test_acc:.4f}"
                )
                if args.save_model:
                    save_model_shares(model, output_root, dataset_name, run_idx)

            row = {
                "dataset_name": dataset_name,
                "experiment": args.experiment,
                "generator": args.generator,
                "num_nodes": num_nodes,
                "target_mean_degree": mean_degree,
                "actual_mean_degree": stats["mean_degree"],
                "actual_max_degree": stats["max_degree"],
                "density": stats["density"],
                "train_runs": args.run,
                "best_val_acc_runs": run_best_vals,
                "best_test_acc_runs": run_best_tests,
                "best_test_acc_mean": float(np.mean(run_best_tests)),
                "best_test_acc_std": float(np.std(run_best_tests)),
            }
            result_rows.append(row)

            print(
                f"[Summary] {dataset_name} "
                f"test_mean={row['best_test_acc_mean']:.4f} "
                f"test_std={row['best_test_acc_std']:.4f}"
            )

    if args.save_results:
        maybe_write_results(output_root, result_rows)


if __name__ == "__main__":
    main()