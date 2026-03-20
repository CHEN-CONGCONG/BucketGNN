#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from ruamel.yaml import YAML
except ImportError:
    YAML = None


class NetworkManager:
    def __init__(self, run_type: str):
        self.run_type = run_type.lower()
        self.script_path = Path(__file__).parent / "throttle.sh"
        self._validate()

    def _validate(self):
        if not self.script_path.exists():
            raise FileNotFoundError(f"Network throttle script not found: {self.script_path}")
        if self.run_type in ("lan", "wan") and shutil.which("tc") is None:
            raise RuntimeError("Missing 'tc' command. Install iproute2 first.")
        if not os.access(self.script_path, os.X_OK):
            self.script_path.chmod(0o755)

    def configure(self):
        if self.run_type == "localhost":
            return
        result = subprocess.run([str(self.script_path), self.run_type], capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Network configuration failed: {err}")

    def cleanup(self):
        if self.run_type == "localhost":
            return
        subprocess.run([str(self.script_path), "del"], capture_output=True, text=True)


def validate_requirements(run_type: str):
    if not shutil.which("make"):
        raise RuntimeError("Missing make command")
    if not shutil.which("g++"):
        raise RuntimeError("Missing g++ command")
    if run_type in ("lan", "wan") and not shutil.which("tc"):
        raise RuntimeError("Missing tc command (iproute2)")


def compile_project(clean_build: bool = False):
    project_dir = Path(__file__).parent.parent
    original_dir = Path.cwd()
    try:
        os.chdir(project_dir)
        if clean_build:
            subprocess.run(["make", "clean"], check=True)
        subprocess.run(["make", "all"], check=True)
    finally:
        os.chdir(original_dir)


def load_yaml(yaml_file: Path):
    if YAML is None:
        raise RuntimeError("Missing ruamel.yaml, please install: pip install ruamel.yaml")
    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    with yaml_file.open("r", encoding="utf-8") as f:
        config = yaml_rt.load(f)
    return config, yaml_rt


def save_yaml(yaml_file: Path, config, yaml_rt):
    with yaml_file.open("w", encoding="utf-8") as f:
        yaml_rt.dump(config, f)


def update_yaml_params(yaml_file: Path, params: dict):
    config, yaml_rt = load_yaml(yaml_file)

    if "dataset" not in config:
        config["dataset"] = {}
    if "properties" not in config["dataset"]:
        config["dataset"]["properties"] = {}
    if "network" not in config["dataset"]:
        config["dataset"]["network"] = {}
    if "graph" not in config["dataset"]:
        config["dataset"]["graph"] = {}
    if "bucket" not in config["dataset"]["graph"]:
        config["dataset"]["graph"]["bucket"] = {}

    for key, value in params.items():
        if key == "hidden_layer_size":
            config["dataset"]["network"]["hidden_layer_size"] = value
        elif key == "position_swap":
            config["dataset"]["position_swap"] = value
        elif key == "sai":
            config["dataset"]["sai"] = value
        elif key == "batch_array_access":
            config["dataset"]["graph"]["batch_array_access"] = value
        elif key == "base_array_size":
            config["dataset"]["graph"]["base_array_size"] = value
        elif key == "mask_neighbor":
            config["dataset"]["graph"]["mask_neighbor"] = value
        elif key == "neighbor_padding_mode":
            config["dataset"]["graph"]["neighbor_padding_mode"] = value
        elif key == "bucket_padding_strategy":
            config["dataset"]["graph"]["bucket"]["strategy"] = value
        elif key == "bucket_merge_mode":
            config["dataset"]["graph"]["bucket"]["merge_mode"] = value
        elif key == "bucket_min_nodes_k":
            config["dataset"]["graph"]["bucket"]["min_nodes_k"] = value
        elif key == "bucket_count":
            config["dataset"]["graph"]["bucket"]["count"] = value
        elif key == "bucket_exp_base":
            config["dataset"]["graph"]["bucket"]["exp_base"] = value
        elif key == "bucket_boundaries":
            config["dataset"]["graph"]["bucket"]["boundaries"] = value
        elif key == "no_of_runs":
            config["dataset"]["network"]["no_of_runs"] = value

    save_yaml(yaml_file, config, yaml_rt)


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def resolve_yaml_files(args) -> list[Path]:
    params_dir = Path(__file__).parent.parent / "parameters"

    yaml_files: list[Path] = []
    if args.yaml_files:
        yaml_files = [Path(p) for p in args.yaml_files]
    elif args.dataset_names:
        yaml_files = [params_dir / f"{name}.yaml" for name in args.dataset_names]
    else:
        yaml_files = sorted(params_dir.glob(f"{args.dataset_prefix}*.yaml"))

    missing = [str(p) for p in yaml_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing YAML files: " + ", ".join(missing))
    if not yaml_files:
        raise FileNotFoundError("No YAML files found for synthetic datasets.")
    return yaml_files


def build_party_command(party_num: int, args, dataset_name: str, output_dir: Path):
    binary_name = Path(__file__).parent.parent / "BucketGNN.out"
    if not binary_name.exists():
        raise FileNotFoundError(f"Binary not found: {binary_name}")

    return [
        str(binary_name),
        str(party_num),
        "files/IP_localhost",
        f"files/key{['A', 'B', 'C'][party_num]}",
        f"files/key{['AC', 'BC', 'AB'][party_num]}",
        f"files/key{['AB', 'AC', 'BC'][party_num]}",
        args.network.upper(),
        dataset_name,
        args.security,
        str(output_dir),
        args.mode,
    ]


def execute_with_network(args, yaml_files: list[Path]):
    project_root = Path(__file__).parent.parent
    net_mgr = NetworkManager(args.run_type)

    print("=" * 80)
    print("MPC Synthetic Inference Runner")
    print(f"Network Type: {args.run_type}")
    print(f"Mode: {args.mode}")
    print(f"Network: {args.network}")
    print(f"YAML files: {len(yaml_files)}")
    print(f"Output Directory: {args.output}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)

    try:
        net_mgr.configure()

        for yaml_path in yaml_files:
            dataset_name = yaml_path.stem
            print(f"\nProcessing synthetic dataset: {dataset_name}")

            base_params = {}
            if args.hidden_layer_size is not None:
                base_params["hidden_layer_size"] = args.hidden_layer_size
            if args.base_array_size is not None:
                base_params["base_array_size"] = args.base_array_size
            if args.sai is not None:
                base_params["sai"] = args.sai
            if args.position_swap is not None:
                base_params["position_swap"] = args.position_swap
            if args.batch_array_access is not None:
                base_params["batch_array_access"] = args.batch_array_access
            if args.mask_neighbor is not None:
                base_params["mask_neighbor"] = args.mask_neighbor
            if args.neighbor_padding_mode is not None:
                base_params["neighbor_padding_mode"] = args.neighbor_padding_mode
            if args.bucket_padding_strategy is not None:
                base_params["bucket_padding_strategy"] = args.bucket_padding_strategy
            if args.bucket_merge_mode is not None:
                base_params["bucket_merge_mode"] = args.bucket_merge_mode
            if args.bucket_min_nodes_k is not None:
                base_params["bucket_min_nodes_k"] = args.bucket_min_nodes_k
            if args.bucket_count is not None:
                base_params["bucket_count"] = args.bucket_count
            if args.bucket_exp_base is not None:
                base_params["bucket_exp_base"] = args.bucket_exp_base
            if args.bucket_boundaries is not None:
                base_params["bucket_boundaries"] = args.bucket_boundaries

            for run_idx in range(args.runs):
                run_params = dict(base_params)
                run_params["no_of_runs"] = run_idx
                print(f"[Run {run_idx}] Update YAML: {yaml_path.name} with {run_params}")
                update_yaml_params(yaml_path, run_params)

                output_dir = (
                    Path(args.output)
                    / dataset_name
                    / args.network
                    / args.mode
                    / args.run_type
                    / f"runs_{run_idx}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                commands = {
                    0: build_party_command(0, args, dataset_name, output_dir),
                    1: build_party_command(1, args, dataset_name, output_dir),
                    2: build_party_command(2, args, dataset_name, output_dir),
                }

                for party, cmd in commands.items():
                    print(f"Party{party}: {' '.join(cmd)}")

                if args.dry_run:
                    continue

                processes = []
                for party, cmd in commands.items():
                    party_name = ["A", "B", "C"][party]
                    log_file = output_dir / f"Party{party_name}.txt"
                    with log_file.open("w") as f:
                        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                        processes.append((party, p))

                for party, p in processes:
                    ret = p.wait()
                    if ret != 0:
                        raise RuntimeError(f"Party {party} failed with code: {ret}")

        print("\nAll synthetic datasets completed.")
    finally:
        net_mgr.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Run MPC inference for synthetic datasets")
    parser.add_argument("--run_type", "-t", required=True, choices=["localhost", "lan", "wan"])
    parser.add_argument("--mode", "-m", default="test", choices=["test", "init", "all", "debug"])
    parser.add_argument("--network", "-n", type=str.lower, default="gcn", choices=["gcn", "sgc", "appnp"])
    parser.add_argument("--runs", "-r", type=int, default=1)
    parser.add_argument("--output", "-o", default="output_ciphertext_inference")

    parser.add_argument("--dataset_names", nargs="+", default=None, help="Synthetic dataset names without .yaml")
    parser.add_argument("--yaml_files", nargs="+", default=None, help="Explicit YAML paths")
    parser.add_argument("--dataset_prefix", default="scalability_", help="Prefix for auto-discovering YAML files")

    parser.add_argument("--sai", dest="sai", action="store_true", default=None)
    parser.add_argument("--no-sai", dest="sai", action="store_false")
    parser.add_argument("--position-swap", dest="position_swap", action="store_true", default=None)
    parser.add_argument("--no-position-swap", dest="position_swap", action="store_false")
    parser.add_argument("--batch-array-access", dest="batch_array_access", action="store_true", default=None)
    parser.add_argument("--no-batch-array-access", dest="batch_array_access", action="store_false")
    parser.add_argument("--mask-neighbor", dest="mask_neighbor", action="store_true", default=None)
    parser.add_argument("--no-mask-neighbor", dest="mask_neighbor", action="store_false")

    parser.add_argument("--hidden_layer_size", type=int, default=None)
    parser.add_argument("--base_array_size", type=int, default=None)
    parser.add_argument("--neighbor_padding_mode", type=str, default=None, choices=["bucket", "random", "secgnn", "none"])
    parser.add_argument("--bucket_padding_strategy", type=str, default=None, choices=["exponential", "equal", "custom"])
    parser.add_argument("--bucket_merge_mode", type=str, default=None, choices=["full_bucket", "partial_fill"])
    parser.add_argument("--bucket_min_nodes_k", type=int, default=None)
    parser.add_argument("--bucket_count", type=int, default=None)
    parser.add_argument("--bucket_exp_base", type=int, default=None)
    parser.add_argument("--bucket_boundaries", type=parse_int_list, default=None)
    parser.add_argument("--security", type=str, default="Semi-honest")

    parser.add_argument("--clean-build", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.runs <= 0:
        print("Error: --runs must be >= 1")
        sys.exit(1)

    if args.yaml_files and (args.dataset_names or args.dataset_prefix):
        pass

    bucket_args_provided = any([
        args.bucket_padding_strategy is not None,
        args.bucket_merge_mode is not None,
        args.bucket_min_nodes_k is not None,
        args.bucket_count is not None,
        args.bucket_exp_base is not None,
        args.bucket_boundaries is not None,
    ])
    if bucket_args_provided and args.neighbor_padding_mode not in (None, "bucket"):
        print("Error: bucket parameters can only be used when --neighbor_padding_mode bucket")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    try:
        yaml_files = resolve_yaml_files(args)
        print("Resolved YAML files:")
        for p in yaml_files:
            print(f"  - {p}")

        validate_requirements(args.run_type)
        if not args.skip_compile:
            compile_project(clean_build=args.clean_build)
        execute_with_network(args, yaml_files)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()