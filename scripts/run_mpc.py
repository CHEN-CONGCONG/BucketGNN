#!/usr/bin/env python3
"""
MPC Secure Inference Runner Script
Allows control over:
  - Network type (localhost, lan, wan)
  - Mode (default: test)
  - Network architecture (default: gcn)
  - Dataset (7 options: cora, citeseer, pubmed, ogbn-arxiv, ogbn-products, amazon-computer, amazon-photo, coauthor-cs, coauthor-physics)
  - Number of runs/inference batches (default: 5)
  - YAML parameters (position_swap, batch_array_access, etc.)
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

try:
    from ruamel.yaml import YAML
except ImportError:
    YAML = None

# Supported datasets for MPC inference
DATASETS = {
    'all7': ['cora', 'citeseer', 'pubmed', 'amazon-computer', 'amazon-photo', 'coauthor-cs', 'coauthor-physics'],
    'all3': ['cora', 'citeseer', 'pubmed'],
    'all4': ['amazon-computer', 'amazon-photo', 'coauthor-cs', 'coauthor-physics'],
    'ablation': ['cora', 'coauthor-physics'],
    'cora': ['cora'],
    'citeseer': ['citeseer'],
    'pubmed': ['pubmed'],
    'ogbn-arxiv': ['ogbn-arxiv'],
    'ogbn-products': ['ogbn-products'],
    'amazon-computer': ['amazon-computer'],
    'amazon-photo': ['amazon-photo'],
    'coauthor-cs': ['coauthor-cs'],
    'coauthor-physics': ['coauthor-physics']
}

class NetworkManager:
    """Network configuration manager for localhost/lan/wan runs."""

    def __init__(self, run_type):
        self.run_type = run_type.lower()
        self.script_path = Path(__file__).parent / 'throttle.sh'
        self._validate()

    def _validate(self):
        if not self.script_path.exists():
            raise FileNotFoundError(f"Network throttle script not found: {self.script_path}")

        if self.run_type in ('lan', 'wan') and shutil.which('tc') is None:
            raise RuntimeError(
                "Missing 'tc' command. Install iproute2 first, e.g. 'sudo apt-get install -y iproute2'"
            )

        if not os.access(self.script_path, os.X_OK):
            print(f"Adding execute permission to {self.script_path}")
            self.script_path.chmod(0o755)

    def configure(self):
        if self.run_type == 'localhost':
            return

        cmd = [str(self.script_path), self.run_type]
        print(f"Applying {self.run_type.upper()} network throttle...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Network configuration failed: {err}")

    def cleanup(self):
        if self.run_type == 'localhost':
            return

        print("Cleaning network throttle...")
        result = subprocess.run([str(self.script_path), 'del'], capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            print(f"Warning: failed to clean network throttle: {err}")

def load_yaml(yaml_file):
    """Load YAML configuration with round-trip parser to preserve comments/layout."""
    if YAML is None:
        raise RuntimeError(
            "Missing ruamel.yaml. Please install it first: pip install ruamel.yaml"
        )

    yaml_rt = YAML(typ='rt')
    yaml_rt.preserve_quotes = True
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml_rt.load(f)
    return config, yaml_rt

def save_yaml(yaml_file, config, yaml_rt):
    """Save YAML configuration while preserving comments/layout."""
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml_rt.dump(config, f)

def update_yaml_params(yaml_file, params):
    """Update specific parameters in YAML file"""
    config, yaml_rt = load_yaml(yaml_file)
    
    if 'dataset' not in config:
        config['dataset'] = {}
    if 'properties' not in config['dataset']:
        config['dataset']['properties'] = {}
    if 'network' not in config['dataset']:
        config['dataset']['network'] = {}
    if 'graph' not in config['dataset']:
        config['dataset']['graph'] = {}
    if 'bucket' not in config['dataset']['graph']:
        config['dataset']['graph']['bucket'] = {}
    
    # Map parameter updates
    for key, value in params.items():
        if key == 'hidden_layer_size':
            config['dataset']['network']['hidden_layer_size'] = value
        elif key == 'position_swap':
            config['dataset']['position_swap'] = value
        elif key == 'sai':
            config['dataset']['sai'] = value
        elif key == 'batch_array_access':
            config['dataset']['graph']['batch_array_access'] = value
        elif key == 'base_array_size':
            config['dataset']['graph']['base_array_size'] = value
        elif key == 'mask_neighbor':
            config['dataset']['graph']['mask_neighbor'] = value
        elif key == 'neighbor_padding_mode':
            config['dataset']['graph']['neighbor_padding_mode'] = value
        elif key == 'bucket_padding_strategy':
            config['dataset']['graph']['bucket']['strategy'] = value
        elif key == 'bucket_merge_mode':
            config['dataset']['graph']['bucket']['merge_mode'] = value
        elif key == 'bucket_min_nodes_k':
            config['dataset']['graph']['bucket']['min_nodes_k'] = value
        elif key == 'bucket_count':
            config['dataset']['graph']['bucket']['count'] = value
        elif key == 'bucket_exp_base':
            config['dataset']['graph']['bucket']['exp_base'] = value
        elif key == 'bucket_boundaries':
            config['dataset']['graph']['bucket']['boundaries'] = value
        elif key == 'no_of_runs':
            config['dataset']['network']['no_of_runs'] = value
    
    save_yaml(yaml_file, config, yaml_rt)

def get_yaml_file(dataset):
    """Get the YAML file path for a dataset"""
    yaml_dir = 'parameters'
    yaml_file = os.path.join(yaml_dir, f"{dataset}.yaml")
    
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")
    
    return yaml_file

def validate_requirements(run_type, require_build_tools=True):
    """Validate system dependencies."""
    if require_build_tools and not shutil.which('make'):
        raise RuntimeError('make command not found. Please install GNU Make first')
    if require_build_tools and not shutil.which('g++'):
        raise RuntimeError('g++ compiler not found. Please install g++ first')
    if run_type in ('lan', 'wan') and not shutil.which('tc'):
        raise RuntimeError(
            'Missing traffic control tool (tc). Please install the iproute2 package: '
            'sudo apt-get update && sudo apt-get install -y iproute2 iperf3 iputils-ping'
        )


def compile_project(clean_build=False):
    """Compile project source code."""
    project_dir = Path(__file__).parent.parent
    original_dir = Path.cwd()

    try:
        print(f"Switching to project directory: {project_dir}")
        os.chdir(project_dir)

        if clean_build:
            print('Cleaning previous build artifacts...')
            subprocess.run(['make', 'clean'], check=True)

        print('Compiling...')
        result = subprocess.run(
            ['make', 'all'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print('Build output:')
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {str(e)}")
        print('Error output:', e.stdout)
        raise RuntimeError('Build error. Please check your Makefile configuration') from e
    except FileNotFoundError as e:
        raise RuntimeError('make command not found. Please ensure build tools are installed') from e
    finally:
        os.chdir(original_dir)
        print(f"Restored working directory to: {original_dir}")


def build_party_command(party_num, args, output_dir):
    """Build command for one MPC party."""
    binary_name = Path(__file__).parent.parent / 'BucketGNN.out'
    if not binary_name.exists():
        raise FileNotFoundError(f'Binary not found: {binary_name}')

    return [
        str(binary_name),
        str(party_num),
        'files/IP_localhost',
        f"files/key{['A', 'B', 'C'][party_num]}",
        f"files/key{['AC', 'BC', 'AB'][party_num]}",
        f"files/key{['AB', 'AC', 'BC'][party_num]}",
        args.network.upper(),
        args.dataset_current,
        args.security,
        str(output_dir),
        args.mode,
    ]


def execute_with_network(args):
    """Execute inference workflow with network configuration."""
    net_mgr = NetworkManager(args.run_type)
    project_root = Path(__file__).parent.parent

    dataset_list = DATASETS[args.dataset]
    print('=' * 80)
    print('MPC Secure Inference Runner')
    print(f'Network Type: {args.run_type}')
    print(f'Mode: {args.mode}')
    print(f'Network: {args.network}')
    print(f"Datasets: {', '.join(dataset_list)}")
    print(f'Runs per dataset: {args.runs}')
    print(f'Output Directory: {args.output}')
    print('=' * 80)

    try:
        net_mgr.configure()

        for dataset in dataset_list:
            print(f"\n{'=' * 80}")
            print(f'Processing dataset: {dataset}')
            print(f"{'=' * 80}")

            yaml_file = get_yaml_file(dataset)
            args.dataset_current = dataset

            base_params = {}
            if args.hidden_layer_size is not None:
                base_params['hidden_layer_size'] = args.hidden_layer_size
            if args.base_array_size is not None:
                base_params['base_array_size'] = args.base_array_size
            if args.sai is not None:
                base_params['sai'] = args.sai
            if args.position_swap is not None:
                base_params['position_swap'] = args.position_swap
            if args.batch_array_access is not None:
                base_params['batch_array_access'] = args.batch_array_access
            if args.mask_neighbor is not None:
                base_params['mask_neighbor'] = args.mask_neighbor
            if args.neighbor_padding_mode is not None:
                base_params['neighbor_padding_mode'] = args.neighbor_padding_mode
            if args.bucket_padding_strategy is not None:
                base_params['bucket_padding_strategy'] = args.bucket_padding_strategy
            if args.bucket_merge_mode is not None:
                base_params['bucket_merge_mode'] = args.bucket_merge_mode
            if args.bucket_min_nodes_k is not None:
                base_params['bucket_min_nodes_k'] = args.bucket_min_nodes_k
            if args.bucket_count is not None:
                base_params['bucket_count'] = args.bucket_count
            if args.bucket_exp_base is not None:
                base_params['bucket_exp_base'] = args.bucket_exp_base
            if args.bucket_boundaries is not None:
                base_params['bucket_boundaries'] = args.bucket_boundaries

            for run_idx in range(args.runs):
                run_params = dict(base_params)
                run_params['no_of_runs'] = run_idx
                print(f"\n[Run {run_idx} ({args.dataset_current} | {args.run_type})] Updating YAML parameters: {run_params}")
                update_yaml_params(yaml_file, run_params)

                output_dir = (
                    Path(args.output)
                    / dataset
                    / args.network
                    / args.mode
                    / args.run_type
                    / f'runs_{run_idx}'
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                commands = {
                    0: build_party_command(0, args, output_dir),
                    1: build_party_command(1, args, output_dir),
                    2: build_party_command(2, args, output_dir),
                }

                processes = []
                for party, cmd in commands.items():
                    print(f"Starting Party{party}: {' '.join(cmd)}")
                    party_name = ['A', 'B', 'C'][party]
                    log_file = output_dir / f"Party{party_name}.txt"
                    print(f"Log file: {(project_root / log_file).resolve()}")

                    with open(log_file, 'w') as f:
                        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                        processes.append((party, p))

                for party, p in processes:
                    ret = p.wait()
                    if ret != 0:
                        raise RuntimeError(f'Party {party} failed with return code: {ret}')

            print(f"\nCompleted inference for dataset: {dataset}")
    finally:
        net_mgr.cleanup()

def main():
    def parse_int_list(value):
        return [int(x.strip()) for x in value.split(',') if x.strip()]

    parser = argparse.ArgumentParser(
        description='MPC Secure Inference Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset inference with LAN network
  python scripts/run_mpc.py --run_type lan -d cora -r 5 --output output_ciphertext
  
  # Multiple datasets with parameters
    python scripts/run_mpc.py --run_type wan -d all7 -r 5 \\
        --sai --position-swap --batch-array-access \\
    --base_array_size 2 --output output_ciphertext
  
  # Localhost with custom hidden layer size
  python scripts/run_mpc.py --run_type localhost -d cora -r 1 \\
    --hidden_layer_size 64 --output output_local
        """
    )
    
    # Network and mode arguments
    parser.add_argument('--run_type', '-t', type=str, required=True, 
                        choices=['localhost', 'lan', 'wan'],
                        help='Network type: localhost, lan, or wan')
    parser.add_argument('-m', '--mode', type=str, default='test',
                        choices=['test', 'init', 'all', 'debug'],
                        help='Running mode (default: test)')
    parser.add_argument('-n', '--network', type=str.lower, default='gcn',
                        choices=['gcn', 'sgc', 'appnp'],
                        help='Network architecture: gcn, sgc, appnp (default: gcn)')
    
    # Dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default='cora',
                        choices=list(DATASETS.keys()),
                        help='Dataset name or dataset group')
    parser.add_argument('-r', '--runs', type=int, default=5,
                        help='Number of runs/inference batches (default: 5)')
    
    # Output arguments
    parser.add_argument('--output', '-o', type=str, default='output_ciphertext_inference',
                        help='Output directory path')
    
    # YAML parameter arguments (boolean flags)
    parser.add_argument('--sai', dest='sai', action='store_true', default=None,
                        help='Enable SAI (secure aggregation initialization)')
    parser.add_argument('--no-sai', dest='sai', action='store_false',
                        help='Disable SAI (secure aggregation initialization)')
    parser.add_argument('--position-swap', dest='position_swap', action='store_true', default=None,
                        help='Enable position swap')
    parser.add_argument('--no-position-swap', dest='position_swap', action='store_false',
                        help='Disable position swap')
    parser.add_argument('--batch-array-access', dest='batch_array_access', action='store_true', default=None,
                        help='Enable batch array access')
    parser.add_argument('--no-batch-array-access', dest='batch_array_access', action='store_false',
                        help='Disable batch array access')
    parser.add_argument('--mask-neighbor', dest='mask_neighbor', action='store_true', default=None,
                        help='Enable neighbor masking')
    parser.add_argument('--no-mask-neighbor', dest='mask_neighbor', action='store_false',
                        help='Disable neighbor masking')
    
    # YAML parameter arguments (values)
    parser.add_argument('--hidden_layer_size', type=int, default=None,
                        help='Hidden layer size')
    parser.add_argument('--base_array_size', type=int, default=None,
                        help='Base array size')
    parser.add_argument('--neighbor_padding_mode', type=str, default=None,
                        choices=['bucket', 'random', 'secgnn', 'none'],
                        help='Neighbor padding mode: bucket, random, secgnn, none')
    parser.add_argument('--bucket_padding_strategy', type=str, default=None,
                        choices=['exponential', 'equal', 'custom'],
                        help='Bucket padding strategy when neighbor_padding_mode=bucket')
    parser.add_argument('--bucket_merge_mode', type=str, default=None,
                        choices=['full_bucket', 'partial_fill'],
                        help='Bucket merge mode when neighbor_padding_mode=bucket')
    parser.add_argument('--bucket_min_nodes_k', type=int, default=None,
                        help='Bucket min nodes k when neighbor_padding_mode=bucket')
    parser.add_argument('--bucket_count', type=int, default=None,
                        help='Bucket count when neighbor_padding_mode=bucket')
    parser.add_argument('--bucket_exp_base', type=int, default=None,
                        help='Bucket exponential base when neighbor_padding_mode=bucket')
    parser.add_argument('--bucket_boundaries', type=parse_int_list, default=None,
                        help='Bucket boundaries list when neighbor_padding_mode=bucket, e.g. 4,8,16,32,168')
    parser.add_argument('--security', type=str, default='Semi-honest',
                        help='Security mode passed to binary (default: Semi-honest)')
    parser.add_argument('--clean-build', action='store_true',
                        help='Run make clean before compile')
    parser.add_argument('--skip-compile', action='store_true',
                        help='Skip compilation and use existing BucketGNN.out binary')
    
    args = parser.parse_args()

    if args.runs <= 0:
        print("Error: --runs must be >= 1")
        sys.exit(1)
    
    # Get dataset list
    if args.dataset not in DATASETS:
        print(f"Error: Unknown dataset '{args.dataset}'")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    bucket_args_provided = any([
        args.bucket_padding_strategy is not None,
        args.bucket_merge_mode is not None,
        args.bucket_min_nodes_k is not None,
        args.bucket_count is not None,
        args.bucket_exp_base is not None,
        args.bucket_boundaries is not None,
    ])
    if bucket_args_provided and args.neighbor_padding_mode not in (None, 'bucket'):
        print("Error: bucket parameters can only be used when --neighbor_padding_mode bucket")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    try:
        validate_requirements(args.run_type, require_build_tools=not args.skip_compile)
        if args.skip_compile:
            print('Skipping compilation (--skip-compile enabled).')
        else:
            compile_project(clean_build=args.clean_build)
        execute_with_network(args)
        print("\n" + "=" * 80)
        print('All datasets processed successfully!')
        print(f'Results saved to: {Path(args.output).resolve()}')
        print('=' * 80)
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
