

import random
import torch
import argparse
import numpy as np
from os import path
import pickle as pkl
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,Amazon, Coauthor
import torch_geometric.transforms as T
from torch_geometric.utils import (
    to_undirected,
    remove_self_loops,
    add_self_loops,
    subgraph,
    dropout_adj,
    degree
)
from tqdm import tqdm

from ogb.nodeproppred import NodePropPredDataset


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(
        self,
        split_type="random",
        train_prop=0.5,
        valid_prop=0.25,
        tr_num_per_class=20,
        val_num_per_class=30,
    ):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == "random_all":
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop
            )
            split_idx = {"train": train_idx,
                "valid": valid_idx, "test": test_idx}
        elif split_type == "random_class":
            train_idx, valid_idx, test_idx = class_rand_splits(
                self.label,
                tr_num_per_class=tr_num_per_class,
                val_num_per_class=val_num_per_class,
            )
            split_idx = {"train": train_idx,
                "valid": valid_idx, "test": test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, sub_dataname=""):
    if dataname in ("cora", "citeseer", "pubmed"):
        print("Loading {} dataset...".format(dataname))
        dataset = load_planetoid_dataset(data_dir, dataname)
    elif dataname in ("ogbn-arxiv", "ogbn-products"):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname in ("amazon-photo", "amazon-computer"):
        dataset = load_amazon_dataset(data_dir, dataname)
    elif dataname in ("coauthor-cs", "coauthor-physics"):
        dataset = load_coauthor_dataset(data_dir, dataname)
    else:
        raise ValueError("Invalid dataname")
    return dataset


def load_planetoid_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(
        root=f"{data_dir}/Planetoid", name=name, transform=transform
    )
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_feat": None,
        "num_nodes": num_nodes,
    }
    dataset.label = label

    return dataset


def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f"{data_dir}/ogb")
    dataset.graph = ogb_dataset.graph
    dataset.graph["edge_index"] = torch.as_tensor(dataset.graph["edge_index"])
    dataset.graph["node_feat"] = torch.as_tensor(dataset.graph["node_feat"])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        print(
            "Shape of train, valid, test: ",
            tensor_split_idx["train"].shape,
            tensor_split_idx["valid"].shape,
            tensor_split_idx["test"].shape,
        )
        print(
            "Proportion of train, valid, test: ",
            tensor_split_idx["train"].shape[0] / dataset.graph["num_nodes"],
            tensor_split_idx["valid"].shape[0] / dataset.graph["num_nodes"],
            tensor_split_idx["test"].shape[0] / dataset.graph["num_nodes"],
        )
        return tensor_split_idx

    dataset.get_idx_split = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset

def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == "amazon-photo":
        torch_dataset = Amazon(
            root=f"{data_dir}/Amazon", name="Photo", transform=transform
        )
    elif name == "amazon-computer":
        torch_dataset = Amazon(
            root=f"{data_dir}/Amazon", name="Computers", transform=transform
        )
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_feat": None,
        "num_nodes": num_nodes,
    }
    dataset.label = label

    return dataset


def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == "coauthor-cs":
        torch_dataset = Coauthor(
            root=f"{data_dir}/Coauthor", name="CS", transform=transform
        )
    elif name == "coauthor-physics":
        torch_dataset = Coauthor(
            root=f"{data_dir}/Coauthor", name="Physics", transform=transform
        )
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_feat": None,
        "num_nodes": num_nodes,
    }
    dataset.label = label

    return dataset

import os

def funcCleanFiles(path="data/"):
    # Clear previous data
    if os.path.exists(path + "plain_data"):
        os.system("rm -rf " + path + "plain_data")
    # Clear all files in the dataset directory, do not clear the secondary directory
    if os.path.exists(path):
        files = os.listdir(path)
        for file in files:
            if os.path.isdir(path + file):
                continue
            os.remove(path + file)
    else:
        os.makedirs(path)


def load_fixed_splits(data_dir, dataset, name, protocol):
    splits_lst = []
    # Normalize to lowercase
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed'] and protocol == 'semi':
        splits = {}
        splits['train'] = torch.as_tensor(dataset.train_idx)
        splits['valid'] = torch.as_tensor(dataset.valid_idx)
        splits['test'] = torch.as_tensor(dataset.test_idx)
        splits_lst.append(splits)
    elif name in ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin']:
        for i in range(10):
            splits_file_path = '{}/geom-gcn/splits/{}'.format(data_dir, name) + '_split_0.6_0.2_'+str(i)+'.npz'
            splits = {}
            with np.load(splits_file_path) as splits_file:
                splits['train'] = torch.BoolTensor(splits_file['train_mask'])
                splits['valid'] = torch.BoolTensor(splits_file['val_mask'])
                splits['test'] = torch.BoolTensor(splits_file['test_mask'])
            splits_lst.append(splits)
    elif name in ['amazon-computer', 'amazon-photo', 'coauthor-cs', 'coauthor-physics']:
            splits = {}
            print("Loading fixed split from:", f'{data_dir}/{name}_split.npz')
            idx = np.load(f'{data_dir}/{name}_split.npz')
            splits['train'] = torch.from_numpy(idx['train'])
            splits['valid'] = torch.from_numpy(idx['valid'])
            splits['test'] = torch.from_numpy(idx['test'])
            splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25):
    """ randomly splits label into train/valid/test splits """
    if len(label.shape) > 1 and label.shape[1] > 1:
        labeled_nodes = torch.arange(label.shape[0])
    else:
        labeled_nodes = torch.where(label != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    print("Shape:", train_idx.shape, valid_idx.shape, test_idx.shape)

    return train_idx, valid_idx, test_idx



def get_dataset(dataset_name, seed=1234, rand_split=False, rand_split_class=False, protocol='semi'):
    fix_seed(seed)
    dataset_defaults = {
            "cora": {"train_prop": 0.2},
            "citeseer": {"train_prop": 0.1},
            "pubmed": {"train_prop": 0.04},
            "ogbn-arxiv": {},
            "ogbn-products": {},
    }

    data_dir = "dataset"
    dataset = load_dataset(data_dir, dataset_name)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if rand_split:
        split_idx_lst = dataset.get_idx_split(
            split_type="random_all", train_prop=dataset_defaults[dataset_name]["train_prop"],
        )
    elif rand_split_class:
        split_idx_lst = dataset.get_idx_split(
            split_type="random_class",
            tr_num_per_class=args.tr_num_per_class,
            val_num_per_class=args.val_num_per_class,
        )
    elif dataset_name in ["ogbn-proteins", "ogbn-arxiv", "ogbn-products"]:
        split_idx_lst = dataset.get_idx_split()
    else:
        if dataset_name in ["cora", "citeseer", "pubmed"] and protocol == "semi":
            split_idx_lst = load_fixed_splits(
                data_dir, dataset, name=dataset_name, protocol=protocol
            )
        elif (
            dataset_name
            in ["chameleon", "squirrel", "film", "cornell", "texas", "wisconsin"]
            and args.protocol == "supervised"
        ):
            split_idx_lst = load_fixed_splits(
                data_dir, dataset, name=dataset_name, protocol=protocol
            )
        else:
            split_idx_lst = load_fixed_splits(
                data_dir, dataset, name=dataset_name, protocol=protocol
            )
        
    dataset.graph["edge_index"], dataset.graph["node_feat"] = dataset.graph[
        "edge_index"
    ], dataset.graph["node_feat"]
    dataset.graph["edge_index"], _ = remove_self_loops(dataset.graph["edge_index"])
    
    return dataset, split_idx_lst


class HandleRSSDataset:
    def __init__(
        self,
        dataset_name,
        save_dir,
        dataset=None,
        split_idx=None,
        max_degree=0,
        mean_degree=0,
    ):
        self.dataset_name = dataset_name
        self.data_path = path.join(save_dir, dataset_name)
        self.dataset = dataset
        if self.dataset is None:
            print(f"Dataset {self.dataset_name} not found.")
            return
        self.split_idx = split_idx
        if self.split_idx is None:
            print(f"Split index not found for {self.dataset_name}.")
            return
        self.train_idx = split_idx["train"]
        self.test_idx = split_idx["test"]
        self.valid_idx = torch.cat([self.train_idx, split_idx["valid"]], dim=-1)

        if not path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.x = self.dataset.graph["node_feat"]
        self.edge_index = self.dataset.graph["edge_index"]
        self.edge_weight = None

        self.n = self.dataset.graph["num_nodes"]
        # infer the number of classes for non one-hot and one-hot labels
        self.c = max(self.dataset.label.max().item() + 1, self.dataset.label.shape[1])
        self.d = self.dataset.graph["node_feat"].shape[1]
        
        self.max_degree = max_degree
        self.mean_degree = mean_degree

    def save_train_data(
        self,
    ):
        print("Saving train data to ---->", self.data_path)
        print("Train index shape: ", self.train_idx.shape)

        if not os.path.exists(path.join(self.data_path, "train")):
            os.makedirs(path.join(self.data_path, "train"))

        paths = {
            "node_data_A": f"train/node_data_A",
            "node_data_B": f"train/node_data_B",
            "node_data_C": f"train/node_data_C",
            "node_labels_A": f"train/node_labels_A",
            "node_labels_B": f"train/node_labels_B",
            "node_labels_C": f"train/node_labels_C",
        }

        if (
            os.path.exists(path.join(self.data_path, paths["node_data_A"]))
            and os.path.exists(path.join(self.data_path, paths["node_data_B"]))
            and os.path.exists(path.join(self.data_path, paths["node_data_C"]))
        ):
            print("Node data files already exist")
        else:
            # Prepare data
            data_A = [self.x[i].tolist() for i in self.train_idx]
            zero_data = [[0] * len(data_A[i]) for i in range(len(data_A))]

            # Save data files
            self._write_file(path.join(self.data_path, paths["node_data_A"]), data_A)
            self._write_file(path.join(self.data_path, paths["node_data_B"]), zero_data)
            self._write_file(path.join(self.data_path, paths["node_data_C"]), zero_data)

        # Save labels
        if (
            os.path.exists(path.join(self.data_path, paths["node_labels_A"]))
            and os.path.exists(path.join(self.data_path, paths["node_labels_B"]))
            and os.path.exists(path.join(self.data_path, paths["node_labels_C"]))
        ):
            print("Node label files already exist")
        else:
            true_label = self._get_label(self.train_idx)
            labels_zero_data = [
                [0] * len(true_label[i]) for i in range(len(true_label))
            ]
            self._write_file(
                path.join(self.data_path, paths["node_labels_A"]), true_label
            )
            self._write_file(
                path.join(self.data_path, paths["node_labels_B"]), labels_zero_data
            )
            self._write_file(
                path.join(self.data_path, paths["node_labels_C"]), labels_zero_data
            )

    def save_test_data(self, clean_adj_list_files=False):
        print("Saving test data to ---->", self.data_path)
        print("Test index shape: ", self.test_idx.shape)

        if not os.path.exists(path.join(self.data_path, "test")):
            os.makedirs(path.join(self.data_path, "test"))

        paths = {
            "node_data_A": f"test/node_data_A",
            "node_data_B": f"test/node_data_B",
            "node_data_C": f"test/node_data_C",
            "node_labels_A": f"test/node_labels_A",
            "node_labels_B": f"test/node_labels_B",
            "node_labels_C": f"test/node_labels_C",
            "adj_list_A": f"test/adj_list_A",
            "adj_list_B": f"test/adj_list_B",
            "adj_list_C": f"test/adj_list_C",
            "edge_weights_A": f"test/edge_weights_A",
            "edge_weights_B": f"test/edge_weights_B",
            "edge_weights_C": f"test/edge_weights_C",
            "test_idx": f"test/test_idx",
            "dataset_info": f"dataset_info.txt",
            "coef_neigh_A": f"node_init/node_init_A",
            "coef_neigh_B": f"node_init/node_init_B",
            "coef_neigh_C": f"node_init/node_init_C",
            "node_state_1_A": f"node_state/node_state_1_A",
            "node_state_1_B": f"node_state/node_state_1_B",
            "node_state_1_C": f"node_state/node_state_1_C",
            "node_state_2_A": f"node_state/node_state_2_A",
            "node_state_2_B": f"node_state/node_state_2_B",
            "node_state_2_C": f"node_state/node_state_2_C",
        }
        # Create directories if they do not exist
        if not os.path.exists(path.join(self.data_path, "node_state")):
            os.makedirs(path.join(self.data_path, "node_state"))
        if not os.path.exists(path.join(self.data_path, "node_init")):
            os.makedirs(path.join(self.data_path, "node_init"))

        if (
            os.path.exists(path.join(self.data_path, paths["node_data_A"]))
            and os.path.exists(path.join(self.data_path, paths["node_data_B"]))
            and os.path.exists(path.join(self.data_path, paths["node_data_C"]))
        ):
            print("Node data files already exist")
        else:
            # all idx
            data_A = [self.x[id].tolist() for id in torch.arange(self.n)]
            zero_data = [[0] * len(data_A[i]) for i in range(len(data_A))]

            # Save data files
            self._write_file(path.join(self.data_path, paths["node_data_A"]), data_A)
            self._write_file(path.join(self.data_path, paths["node_data_B"]), zero_data)
            self._write_file(path.join(self.data_path, paths["node_data_C"]), zero_data)

        # Save labels
        if (
            os.path.exists(path.join(self.data_path, paths["node_labels_A"]))
            and os.path.exists(path.join(self.data_path, paths["node_labels_B"]))
            and os.path.exists(path.join(self.data_path, paths["node_labels_C"]))
        ):
            print("Node label files already exist")
        else:
            # all idx
            true_label = self._get_label(torch.arange(self.n))
            labels_zero_data = [
                [0] * len(true_label[i]) for i in range(len(true_label))
            ]
            self._write_file(
                path.join(self.data_path, paths["node_labels_A"]), true_label
            )
            self._write_file(
                path.join(self.data_path, paths["node_labels_B"]), labels_zero_data
            )
            self._write_file(
                path.join(self.data_path, paths["node_labels_C"]), labels_zero_data
            )

        if (
            os.path.exists(path.join(self.data_path, paths["adj_list_A"]))
            and os.path.exists(path.join(self.data_path, paths["adj_list_B"]))
            and os.path.exists(path.join(self.data_path, paths["adj_list_C"]))
            and not clean_adj_list_files
        ):
            print("Adjacency list files already exist")
            adj_list, edge_weight = self._build_adjacency()
        else:
            # Remove existing files; skip if they do not exist
            if os.path.exists(path.join(self.data_path, paths["adj_list_A"])):
                os.remove(path.join(self.data_path, paths["adj_list_A"]))
            if os.path.exists(path.join(self.data_path, paths["adj_list_B"])):
                os.remove(path.join(self.data_path, paths["adj_list_B"]))
            if os.path.exists(path.join(self.data_path, paths["adj_list_C"])):
                os.remove(path.join(self.data_path, paths["adj_list_C"]))
            if os.path.exists(path.join(self.data_path, paths["edge_weights_A"])):
                os.remove(path.join(self.data_path, paths["edge_weights_A"]))
            if os.path.exists(path.join(self.data_path, paths["edge_weights_B"])):
                os.remove(path.join(self.data_path, paths["edge_weights_B"]))
            if os.path.exists(path.join(self.data_path, paths["edge_weights_C"])):
                os.remove(path.join(self.data_path, paths["edge_weights_C"]))
            if os.path.exists(path.join(self.data_path, "edge_info_plaintext")):
                for file in os.listdir(
                    path.join(self.data_path, "edge_info_plaintext")
                ):
                    os.remove(
                        os.path.join(
                            path.join(self.data_path, "edge_info_plaintext"), file
                        )
                    )

            adj_list, edge_weight = self._build_adjacency()
            adj_zero_data = [[0] * len(adj_list[i]) for i in range(len(adj_list))]
            edge_zero_data = [
                [0] * len(edge_weight[i]) for i in range(len(edge_weight))
            ]

            # Save adjacency list files
            self._write_file(path.join(self.data_path, paths["adj_list_A"]), adj_list)
            self._write_file(
                path.join(self.data_path, paths["adj_list_B"]), adj_zero_data
            )
            self._write_file(
                path.join(self.data_path, paths["adj_list_C"]), adj_zero_data
            )

            # Save edge weight files
            self._write_file(
                path.join(self.data_path, paths["edge_weights_A"]), edge_weight
            )
            self._write_file(
                path.join(self.data_path, paths["edge_weights_B"]), edge_zero_data
            )
            self._write_file(
                path.join(self.data_path, paths["edge_weights_C"]), edge_zero_data
            )

        # Save idx
        if os.path.exists(path.join(self.data_path, paths["test_idx"])):
            print("Test idx files already exist")
        else:
            print(f"Saving test idx")
            self._save_idx(self.test_idx, path.join(self.data_path, paths["test_idx"]))

        # Save dataset info
        with open(path.join(self.data_path, paths["dataset_info"]), "w") as f:
            f.write(f"Number of classes: {self.c}\n")
            f.write(f"Number of node features: {self.d}\n")
            f.write(f"Number of nodes: {self.n}\n")
            f.write(f"Train index shape: {self.train_idx.shape}\n")
            f.write(f"Valid index shape: {self.valid_idx.shape}\n")
            f.write(f"Test index shape: {self.test_idx.shape}\n")
            f.write(
                f"Max degree (not including self loops): {(int(self.max_degree))}\n"
            )
            f.write(f"Mean degree: {self.mean_degree}\n")

        # Save node init
        if (
            os.path.exists(path.join(self.data_path, paths["coef_neigh_A"]))
            and os.path.exists(path.join(self.data_path, paths["coef_neigh_B"]))
            and os.path.exists(path.join(self.data_path, paths["coef_neigh_C"]))
        ):
            print("Node init files already exist")
            sw_reci, coef_neigh = self._compute_node_init(adj_list, edge_weight)
        else:
            sw_reci, coef_neigh = self._compute_node_init(adj_list, edge_weight)

            coef_zero_data = [[0] * len(coef_neigh[i]) for i in range(len(coef_neigh))]
            self._write_file(path.join(self.data_path, paths["coef_neigh_A"]), coef_neigh)
            self._write_file(path.join(self.data_path, paths["coef_neigh_B"]), coef_zero_data)
            self._write_file(path.join(self.data_path, paths["coef_neigh_C"]), coef_zero_data)

        # Save node state
        if (
            os.path.exists(path.join(self.data_path, paths["node_state_1_A"]))
            and os.path.exists(path.join(self.data_path, paths["node_state_1_B"]))
            and os.path.exists(path.join(self.data_path, paths["node_state_1_C"]))
        ):
            print("Node state files already exist")
            # If cached files exist, load them directly
            node_state_1 = [torch.zeros_like(x) for x in self.x]
            with open(path.join(self.data_path, paths["node_state_1_A"]), "r") as f:
                for i in range(self.n):
                    # Read each line using spaces as separators
                    line = f.readline().strip().split()
                    node_state_1[i] = torch.tensor([float(x) for x in line])
        else:
            node_state_1 = self._compute_node_state(
                adj_list, edge_weight, sw_reci, coef_neigh, self.x
            )
            node_state_zero = [
                [0] * len(node_state_1[i]) for i in range(len(node_state_1))
            ]
            self._write_file(
                path.join(self.data_path, paths["node_state_1_A"]), node_state_1
            )
            self._write_file(
                path.join(self.data_path, paths["node_state_1_B"]), node_state_zero
            )
            self._write_file(
                path.join(self.data_path, paths["node_state_1_C"]), node_state_zero
            )

        # Save node state 2
        if os.path.exists(path.join(self.data_path, paths["node_state_2_A"])) and os.path.exists(path.join(self.data_path, paths["node_state_2_B"])) and os.path.exists(path.join(self.data_path, paths["node_state_2_C"])):
            print("Node state 2 files already exist")
        else:
            node_state_2 = self._compute_node_state(adj_list, edge_weight, sw_reci, coef_neigh, node_state_1)
            node_state_zero = [[0] * len(node_state_2[i]) for i in range(len(node_state_2))]
            self._write_file(path.join(self.data_path, paths["node_state_2_A"]), node_state_2)
            self._write_file(path.join(self.data_path, paths["node_state_2_B"]), node_state_zero)
            self._write_file(path.join(self.data_path, paths["node_state_2_C"]), node_state_zero)

    def _save_idx(self, idx, save_path):
        if os.path.exists(save_path):
            return
        progress_bar = tqdm(total=len(idx), desc=f"Writing to {save_path}")
        with open(save_path, "w") as f:
            for i in idx:
                f.write(str(i.item()) + "\n")
                progress_bar.update(1)
        progress_bar.close()

    def _write_file(self, path, data):
        if os.path.exists(path):
            return

        progress_bar = tqdm(total=len(data), desc=f"Writing to {path}")
        with open(path, "w") as f:
            for line in data:
                line = [x.item() if hasattr(x, "item") else x for x in line]
                f.write(" ".join(map(str, line)) + "\n")
                progress_bar.update(1)
        progress_bar.close()

    def _build_adjacency(self, ):
        path_adj_list = path.join(self.data_path, "edge_info_plaintext", "adj_list.pkl")
        path_edge_weight = path.join(
            self.data_path, "edge_info_plaintext", "edge_weight.pkl"
        )
        if not os.path.exists(path.join(self.data_path, "edge_info_plaintext")):
            os.makedirs(path.join(self.data_path, "edge_info_plaintext"))


        # If cache files exist, load them directly
        if (
            os.path.exists(path_adj_list)
            and os.path.exists(path_edge_weight)
        ):
            with open(path_adj_list, "rb") as f:
                adj_list = pkl.load(f)
            with open(path_edge_weight, "rb") as f:
                edge_weight = pkl.load(f)

            print(f"Loaded adjacency list from {path_adj_list}")
            return adj_list, edge_weight

        # Initialize adjacency list and edge weights
        adj_list = []
        edge_weight = []

        progress_bar = tqdm(total=self.n, desc="Building adjacency list")
        for i in range(self.n):
            neighbors = self.edge_index[1][self.edge_index[0] == i].tolist()
            edge_weight.append([1] * len(neighbors))
                        
            adj_list.append(neighbors)
            progress_bar.update(1)

        progress_bar.close()

        # Save results to files
        print(f"Saving adjacency list to {path_adj_list}")
        print(f"Saving edge weight to {path_edge_weight}")
        with open(path_adj_list, "wb") as f:
            pkl.dump(adj_list, f)
        with open(path_edge_weight, "wb") as f:
            pkl.dump(edge_weight, f)

        return adj_list, edge_weight


    def _get_label(
        self,
        idx,
    ):
        label = self.dataset.label[idx]
        device = label.device

        # Replace invalid label -1 with 0
        label = torch.where(label >= 0, label, torch.tensor(0).to(device))

        if label.shape[1] == 1:
            true_label = F.one_hot(label.to(torch.int64), self.c).squeeze(1)
        else:
            true_label = label

        return true_label

    def _compute_node_init(self, adj_list, edge_weight):
        # Compute sw, sw_reci, and sw_inv_sqrt
        sw = np.zeros(self.n)
        sw_reci = np.zeros(self.n)
        sw_inv_sqrt = np.zeros(self.n)
        for i in range(self.n):
            sw[i] = sum(edge_weight[i]) + 1
            sw_reci[i] = 1.0 / sw[i]
            sw_inv_sqrt[i] = 1.0 / np.sqrt(sw[i])

        # Compute coef_neigh
        progress_bar = tqdm(total=self.n, desc="Computing coef_neigh")
        coef_neigh = [None] * self.n
        for node_i in range(self.n):
            coef_neigh[node_i] = [None] * len(adj_list[node_i])

            for j, node_j, weight in zip(
                range(len(adj_list[node_i])), adj_list[node_i], edge_weight[node_i]
            ):
                # Compute sw_inv_sqrt[i] * weight * sw_inv_sqrt[j]
                coef_neigh[node_i][j] = (
                    sw_inv_sqrt[node_i] * weight * sw_inv_sqrt[node_j]
                )
            progress_bar.update(1)
        progress_bar.close()

        return sw_reci, coef_neigh

    def _compute_node_state(
        self, adj_list, edge_weight, sw_reci, coef_neigh, pre_state
    ):
        node_state = [torch.zeros_like(x) for x in self.x]

        # Compute node_state = 1/sw_i * input_i
        for i in range(self.n):
            node_state[i] = sw_reci[i] * pre_state[i]

            # Compute edge_weight * sw_inv_sqrt[i] * sw_inv_sqrt[j] * input_j
            for j, node_j, weight in zip(
                range(len(adj_list[i])), adj_list[i], edge_weight[i]
            ):
                node_state[i] += coef_neigh[i][j] * pre_state[node_j]

        return node_state
    
if  __name__ == "__main__":
    # Use for ogbn-arxiv
    parser = argparse.ArgumentParser(
            description="Process dataset and save RSS shares",
            formatter_class=argparse.RawTextHelpFormatter,
        )
    parser.add_argument(
        "--dataset", "-d", help="dataset, cora, citeseer, pubmed, ogbn-arxiv, ogbn-products", default="cora"
    )
    parser.add_argument(
        "--seed", help="random seed", default=1234
    )
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--tr_num_per_class', type=int, default=20, help='training nodes randomly selected')
    parser.add_argument('--val_num_per_class', type=int, default=30, help='valid nodes randomly selected')
    parser.add_argument('--train_prop', type=float, default=.5,
                            help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                            help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                            help='protocol for splitting dataset, semi or supervised')
    args = parser.parse_args()

    print (f"Args: {args}")

    dataset, split_idx_lst = get_dataset(dataset_name=args.dataset, seed=args.seed, rand_split=args.rand_split, rand_split_class=args.rand_split_class, protocol=args.protocol)
    
    # Save Dataset
    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed'):
        split_idx = split_idx_lst[0]

        _, dst = dataset.graph["edge_index"]
        _degree = degree(dst, num_nodes=dataset.graph["num_nodes"])
        max_degree = max(_degree)
        mean_degree = _degree.mean()
        round_mean_degree = int(torch.round(mean_degree))
        print(f"Max degree: {max_degree}")
        print(f"Mean degree: {mean_degree}, round mean degree: {round_mean_degree}")
        print(
            f"Degree < round mean degree: {(_degree <= round_mean_degree).sum()}, proportion: {(_degree <= round_mean_degree).sum() / len(_degree)}"
        )
        handler = HandleRSSDataset(dataset_name=args.dataset, save_dir='../data', dataset=dataset, split_idx=split_idx,
                           max_degree=max_degree, mean_degree=mean_degree)
        handler.save_train_data()
        handler.save_test_data(clean_adj_list_files=False)
    else:
        # handle_data = HandleData(dataset, split_idx_lst)
        pass