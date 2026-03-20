### Warning
---
This codebase is released solely as a reference for other developers, as a proof-of-concept, and for benchmarking purposes. Our code backend is based on [Falcon](https://github.com/snwagh/falcon-public).


### Requirements
---
* The code should work on most Linux distributions (It has been developed and tested with [Ubuntu](http://www.ubuntu.com/) 16.04 and 18.04).

* **Required packages for FastGNN:**
  * [`g++`](https://packages.debian.org/testing/g++)
  * [`make`](https://packages.debian.org/testing/make)
  * [`libssl-dev`](https://packages.debian.org/testing/libssl-dev)
  * [`libyaml-cpp-dev`](https://packages.debian.org/testing/libyaml-cpp-dev)

  Install these packages with your favorite package manager, e.g, `sudo apt-get update && sudo apt-get install -y g++ make libssl-dev libyaml-cpp-dev`.

#### Repository Structure

* `files/`    - Shared keys, IP addresses and data files.
* `files/preload`    - Contains data for pretrained network. The networks can be generated using `scripts` and functions in `secondary.cpp`
* `lib_eigen/`    - [Eigen library](http://eigen.tuxfamily.org/) for faster matrix multiplication.
* `src/`    - Source code.
* `util/` - Dependencies for AES randomness.

#### Building the code

To build BucketGNN, run the following commands:

```
cd BucketGNN
make all -j$(nproc)
```

#### Running the code

To run the code, simply choose one of the following options: 

* `make`: Prints all the possible makefile options.
* `make terminal`: Runs the 3PC code on localhost with output from $P_0$ printed to standard output.
* `make file`: Runs the 3PC code on localhost with output from $P_0$ printed to a file (in `output/*`)
* `make command_file`: Runs the 3PC code on localhost with output from $P_0$ printed to a file (in `output/*`)
* `make valg`: Useful for debugging the code for set faults. Note that the -03 optimization flag needs to be suppressed (toggle lines 53, 54 in `makefile`)
* `make command`: Enables running a specific network, dataset, adversarial model, and run type (localhost/LAN/WAN) specified through the `makefile`. This takes precedence over choices in the `src/main.cpp` file.
* To run the code over tmux over multiple terminals, `make zero`, `make one`, `make two` and `make three` come in handy.
* Finally, the `makefile` (line 4-17) contains the descriptions of the arguments accepted by the executable.

#### Errors and Issues
If there are compile/installation/runtime errors, please create git issues. Some of the common errors and their resolutions are listed below: 
* `cannot find` something error: This would be a linker error, make sure your code has access to the right paths in the `$PATH` environment variable.
* `Binding error`: There is probably an execution running in the background. Kill the process with `pkill FastGNN.out` (this happens since the socket ports are hardcoded in the program and a number of the makefile commands run code in the background and `Ctrl+c` only kills the Party 0 code)
* `Bad alloc`: This is probably a memory issue. If you're running one of the larger networks (AlexNet, VGG16) on a laptop, try running them on a server/cluster/machine with more memory.

# Network Configuration
In order to simulate LAN, WAN environment, we can use the following command. LAN and WAN corresponded to 10Gbps/400Mbps bandwidth and 0.3ms/40ms round trip latency, respectively.

`sudo apt-get update && sudo apt-get install -y iproute2 iperf3 iputils-ping`.

> During the first run, you may encounter the "RTNETLINK answers: No such file or directory" error. You can safely ignore it.
```bash
# Authorization, only needs to be executed once.
chmod 755 throttle.sh

# Simulate LAN
./throttle.sh lan

# Simulate WAN
./throttle.sh wan

# Delete configuration
./throttle.sh del
```

# Dataset and Model Preparation
- Python==3.10.12 (recommendation)
  - Windows:
```bash
# cuda 12.1
conda create -n bucketgnn python=3.10.12
conda activate bucketgnn
pip install torch==2.2.1 torch_geometric==2.5.0 numpy==1.26.3 tqdm==4.66.1

# check torch version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install googledrivedownloader==0.4
pip install ogb ruamel.yaml openpyxl
```
  - Mac
```bash
  # mps
  conda create -n bucketgnn python=3.10.12
  conda activate bucketgnn
  pip install torch==2.2.1 torch_geometric==2.5.0 numpy==1.26.3 tqdm==4.66.1

  # mac, check
  python -c "import torch; print(torch.backends.mps.is_available())"
  python -c "import torch; print(torch.backends.mps.is_built())"

  pip --no-cache-dir install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cpu.html
  pip install googledrivedownloader==0.4
  pip install ogb ruamel.yaml openpyxl

```


# Running the code
## Run All Experiments
```bash bash
bash run_all.sh
```

## Additional Run Commands

You can also run individual workflows directly:

```bash
# Prepare datasets and pretrained models
bash run_data_preparation.sh

# Run plaintext training
bash run_training_plaintext.sh

# Run synthetic training
bash run_training_synthetic.sh

# Run synthetic MPC inference (scalability experiments)
bash run_synthetic_mpc.sh
```

For `run_synthetic_mpc.sh`, you can override runtime options with environment variables:

```bash
RUN_TYPE=localhost MODE=test NETWORK=gcn RUNS=5 \
OUTPUT_DIR=output_ciphertext_inference_synthetic \
bash run_synthetic_mpc.sh
```

Optional flags:

```bash
DRY_RUN_FLAG=1 bash run_synthetic_mpc.sh
CLEAN_BUILD_FLAG=1 bash run_synthetic_mpc.sh
SKIP_COMPILE_FLAG=1 bash run_synthetic_mpc.sh
```