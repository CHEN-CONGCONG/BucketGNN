import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import os
import numpy as np
from torch_geometric.datasets import Planetoid
import argparse
import random
from dataset import get_dataset 



class Net(torch.nn.Module):
    def __init__(self, feature_dim, label_dim, hidden_dim, dropout=0.8):
        super(Net, self).__init__()
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

    def forward(self, data):
        x = data.graph["node_feat"]
        edge_index = data.graph["edge_index"]

        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)

        return F.log_softmax(x, dim=1)


class Model:
    def __init__(self, dataset, lr=0.01, weight_decay=5e-4, epochs=200, hidden_dim=64, dropout=0.8, device_option="0", run_idx=0, seed=1234):
        self.epochs = epochs
        self.dataset_name = dataset
        self.run_idx = run_idx
        self.dataset, split_idx = get_dataset(dataset, seed=seed)
        if dataset in ["cora", "citeseer", "pubmed",
                "amazon-computer", "amazon-photo", "coauthor-cs", "coauthor-physics"]:
            split_idx = split_idx[0]

        self.feature_dim = self.dataset.graph["node_feat"].shape[1]
        self.label_dim = max(self.dataset.label.max().item() + 1, self.dataset.label.shape[1])

        self.device = resolve_device(device_option)
        print(f"Using device: {self.device}")
        
        self.model = Net(self.feature_dim, self.label_dim, hidden_dim, dropout=dropout).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        # Move dataset to device
        self.dataset.graph["node_feat"] = self.dataset.graph["node_feat"].to(self.device)
        self.dataset.graph["edge_index"] = self.dataset.graph["edge_index"].to(self.device)
        self.dataset.label = self.dataset.label.to(self.device)
        
        # Redefine the training set, validation set, and test set
        self.train_idx = split_idx["train"].to(self.device)
        self.test_idx = split_idx["test"].to(self.device)
        self.valid_idx = torch.cat([self.train_idx, split_idx["valid"].to(self.device)], dim=-1)

        print("Train Node: ", self.train_idx.shape[0])
        print("Test Node: ", self.test_idx.shape[0])
        print("Valid Node: ", self.valid_idx.shape[0])

        self.accuracy = 0

    def train(self,):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(self.dataset)
            true_label = self.dataset.label.squeeze(1).long()
            loss = nn.NLLLoss()(out[self.train_idx], true_label[self.train_idx])

            loss.backward()
            self.optimizer.step()

            self.model.eval()
            _, pred = self.model(self.dataset).max(dim=1)
            true_label_= true_label[self.valid_idx]
            correct = float(
                pred[self.valid_idx].eq(true_label_).sum().item()
                )
            acc = correct / self.valid_idx.shape[0]

            print("Run: {}, Epoch: {:02d} loss: {:.4f} val_acc: {:.4f}".format(self.run_idx + 1, epoch, loss, acc))

    def test(self):
        self.model.eval()
        _, pred = self.model(self.dataset).max(dim=1)
        true_label = self.dataset.label.squeeze(1).long()
        correct = float(
            pred[self.test_idx].eq(true_label[self.test_idx]).sum().item()
        )
        acc = correct / self.test_idx.shape[0]
        self.accuracy = acc
        print(f"Run: {self.run_idx + 1}, Test Accuracy: {acc:.4f}")

    def save_model(self, filename="gcn", run_idx=0):
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        
        output_dir = "../data/" + self.dataset_name + "/network/"
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, filename, f"run_{run_idx}")
        os.makedirs(model_path, exist_ok=True)
        print(f"Saving model to {model_path}")

        # Remove existing model files under this path
        for file in os.listdir(model_path):
            file_path = os.path.join(model_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # define parameters
        parameters = {
            "fc_weight_1": self.model.gcn1.lin.weight.data.cpu().numpy(),
            "fc_bias_1": self.model.gcn1.bias.data.cpu().numpy(),
            "fc_weight_2":  self.model.gcn2.lin.weight.data.cpu().numpy(),
            "fc_bias_2": self.model.gcn2.bias.data.cpu().numpy(),
        }

        # save parameters
        for name, param in parameters.items():
            param_data = param.tolist() if isinstance(param, np.ndarray) else param
            zero_data = np.zeros_like(param).tolist() if isinstance(param, np.ndarray) else param

            write_model_file(os.path.join(model_path, f"{name}_A"), param_data)
            write_model_file(os.path.join(model_path, f"{name}_B"), zero_data)
            write_model_file(os.path.join(model_path, f"{name}_C"), zero_data)
        return os.path.join(output_dir, filename)


def write_summary_info(model_root_path, run_accuracies, args, dataset_name, model):
    mean_acc = float(np.mean(run_accuracies)) if len(run_accuracies) > 0 else 0.0
    std_acc = float(np.std(run_accuracies)) if len(run_accuracies) > 0 else 0.0

    with open(os.path.join(model_root_path, "network_info.txt"), "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write("Method: gcn\n")
        f.write(f"Runs: {len(run_accuracies)}\n")
        f.write(f"Epochs per run: {int(args.epochs)}\n")
        f.write(f"Hidden channels: {model.gcn1.out_channels}\n")
        f.write(f"Dropout: {model.dropout}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Run accuracies: {', '.join([f'{acc:.6f}' for acc in run_accuracies])}\n")
        f.write(f"Accuracy mean: {mean_acc:.6f}\n")
        f.write(f"Accuracy std: {std_acc:.6f}\n")
        f.write(f"Network: \n{model}\n")


def resolve_dropout(dataset_name, user_dropout):
    if user_dropout is not None:
        return float(user_dropout)
    if dataset_name in ["amazon-computer", "amazon-photo"]:
        return 0.8
    return 0.5


def resolve_device(device_option):
    if device_option is None:
        return torch.device("cpu")

    option = str(device_option).strip().lower()
    if option == "":
        return torch.device("cpu")

    if option == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if option.isdigit():
        gpu_index = int(option)
        if gpu_index < torch.cuda.device_count():
            return torch.device(f"cuda:{gpu_index}")
        return torch.device("cpu")

    if option.startswith("cuda"):
        if option == "cuda":
            return torch.device("cuda:0")
        if option.startswith("cuda:"):
            index_text = option.split(":", 1)[1]
            if index_text.isdigit() and int(index_text) < torch.cuda.device_count():
                return torch.device(option)
        return torch.device("cpu")

    return torch.device("cpu")


def run_experiment(dataset_name, hidden_dim, args):
    run_accuracies = []
    model_root_path = None
    last_model = None
    dropout = resolve_dropout(dataset_name, args.dropout)

    for run_idx in range(int(args.run)):
        # Set different random seed for each run to get different results
        seed = int(args.seed) + run_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Disable cuDNN to ensure reproducibility with different seeds
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"\n{'='*50}")
        print(f"Run {run_idx + 1}/{int(args.run)} with seed {seed}")
        print(f"{'='*50}")
        
        model = Model(
            dataset_name,
            epochs=int(args.epochs),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            hidden_dim=int(hidden_dim),
            dropout=dropout,
            device_option=args.device,
            run_idx=run_idx,
            seed=seed
        )
        model.train()
        model.test()
        run_accuracies.append(model.accuracy)
        last_model = model

        if args.save_model:
            model_root_path = model.save_model(run_idx=run_idx)

    if args.save_model and model_root_path is not None and last_model is not None:
        write_summary_info(model_root_path, run_accuracies, args, dataset_name, last_model.model)

def write_model_file(path, data):
        with open(path, "w") as f:
            # Check if data is 1D or 2D and format accordingly
            if isinstance(data[0], list):  # 2D data
                for line in data:
                    line = [x.item() if hasattr(x, "item") else x for x in line]
                    f.write(" ".join(map(str, line)) + "\n")
            else:  # 1D data
                data = [x.item() if hasattr(x, "item") else x for x in data]
                f.write(" ".join(map(str, data)) + "\n")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train GCN model, and save the model', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', '-d', help='dataset, default all3', default="all3")
    parser.add_argument('--epochs', '-e', help='epochs, default 500', default=500)
    parser.add_argument('--learning_rate', '-lr', help='learning rate, default 0.2', default=0.01)
    parser.add_argument('--weight_decay', '-wd', help='weight decay, default 5e-4', default=5e-4)
    parser.add_argument('--hidden_dim', '-hd', help='hidden dim, default 64', default=64)
    parser.add_argument('--dropout', '-do', type=float, help='dropout, default: auto (amazon:0.8, others:0.5)', default=None)
    parser.add_argument('--device', '-dv', help='device: 0/1/... or cpu/cuda:0, default 0, fallback cpu', default="0")
    parser.add_argument('--run', '-r', help='number of runs, default 1', default=5)
    parser.add_argument('--seed', help='random seed, default 1234', default=1234)
    parser.add_argument('--save_model', '-s', help='save model, default True', default=True)
    args = parser.parse_args()

    # Cora, Citeseer, Pubmed
    if args.dataset == "all3":
        for dataset in ["cora", "citeseer", "pubmed"]:
            run_experiment(dataset, 16, args)
    elif args.dataset == "all4":
        for dataset in ["amazon-computer", "amazon-photo", "coauthor-cs", "coauthor-physics"]:
            run_experiment(dataset, 64, args)
    elif args.dataset == "all7":
        for dataset in ["cora", "citeseer", "pubmed"]:
            run_experiment(dataset, 16, args)
        for dataset in ["amazon-computer", "amazon-photo", "coauthor-cs", "coauthor-physics"]:
            run_experiment(dataset, 64, args)
    else:
        run_experiment(args.dataset, args.hidden_dim, args)