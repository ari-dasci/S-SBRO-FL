from copy import deepcopy

import torch
import numpy as np
import random

from flex.pool import select_nodes
from flex.common import load_flex_model
import torch.nn as nnf
from utils import SeedManager


DATASET_NAME = 'fashionmnist'  # Change this to 'cifar', 'svhn', 'emnist' as needed
SAVE_DIR_BASE      = f"../result/supplementary_Lincbe3_test_{DATASET_NAME}/"
price_test = False

if DATASET_NAME == 'emnist':
    INITIAL_MODEL_PATH = "../result/normal/emnist_1008_1515/EMNIST_CNN.pkl"
    done_result_file   = "../result/normal/emnist_1008_1515/result_final.json"
elif DATASET_NAME == 'cifar':
    INITIAL_MODEL_PATH = "../result/normal/cifar_1009_0812/CIFAR10_CNN.pkl"
    done_result_file   = "../result/normal/cifar_1009_0812/result_final.json"
elif DATASET_NAME == 'fashionmnist':
    INITIAL_MODEL_PATH = "../result/normal/fashionmnist_1009_0004/FashionMNIST_CNN.pkl"
    done_result_file   = "../result/normal/fashionmnist_1009_0004/result_final.json"
elif DATASET_NAME == 'svhn':
    INITIAL_MODEL_PATH = "../result/normal/svhn_1008_2055/SVHN_CNN.pkl"
    done_result_file   = "../result/normal/svhn_1008_2055/result_final.json"



def get_experiment_data_params(group_name):
    noise_params = {
        'client_flip_ratio': [0.2, 0.2, 0.2, 0.2],
        'label_flip_ratio': [0.9, 0.8, 0.7, 0.6]
    }
    if group_name == 'emnist':
        data_params = {
            'data_name': "emnist_letters",
            'model_name': "EMNIST_CNN",
            'optimizer': "sgd",
            'initial_lr': 0.01,
            'final_lr': 0.001,
            'total_rounds': 300,
            'batch_size': 16,
            'training_size': 10000,
            'validation_size': 2000
        }
    elif group_name == "cifar":
        data_params = {
            'data_name': "cifar10",
            'model_name': "CIFAR10_CNN",
            'optimizer': "sgd",
            'initial_lr': 0.01,
            'final_lr': 0.001,
            'total_rounds': 300,
            'batch_size': 16,
            'training_size': 10000,
            'validation_size': 2000
        }
    elif  group_name == 'fashionmnist':
        data_params = {
            'data_name': "fashionmnist",
            'model_name': "FashionMNIST_CNN",
            'optimizer': "sgd",
            'initial_lr': 0.01,
            'final_lr': 0.001,
            'total_rounds': 300,
            'batch_size': 16,
            'training_size': 10000,
            'validation_size': 2000
        }
    elif group_name == 'svhn':
        data_params = {
            'data_name': "svhn",
            'model_name': "SVHN_CNN",
            'optimizer': "sgd",
            'initial_lr': 0.1,
            'final_lr': 0.001,
            'total_rounds': 300,
            'batch_size': 16,
            'training_size': 10000,
            'validation_size': 2000
        }
    else:
        raise ValueError(f"Unknown dataset name: {group_name}")
    return {**data_params, **noise_params}
def generate_experiment_parameters(dataset_name):
    group_params = get_experiment_data_params(dataset_name)
    common_params = {
        'seed': 1,
        'eachclient_epoch': 1,
        'ω': 10,
        'ψ': 5,
        'α': 0.15,
        'β': 0.3,
        'γ': 1,
        'n_nodes': 40,
        'Budget': 45,
        'Bid_mean': 10,
        'Bid_stddev': 1,
        'n_rounds': 300
    }
    return {**common_params, **group_params}

experiment_parameters = generate_experiment_parameters(DATASET_NAME)

import os
if price_test:
    experiment_parameters['save_dir'] = os.path.join(SAVE_DIR_BASE, 'price_test')
else:
    experiment_parameters['save_dir'] = os.path.join(SAVE_DIR_BASE, 'normal')
from types import SimpleNamespace
exparam = SimpleNamespace(**experiment_parameters)

# Create directory if it does not exist
import datetime
date_str = datetime.datetime.now().strftime('%m%d_%H%M')
file_path = os.path.join(exparam.save_dir, '_'.join([DATASET_NAME, date_str]))
from flex.common import create_directory
create_directory(file_path)

# Save the parameters
import yaml
with open(os.path.join(file_path, 'params.yaml'), 'w') as f:
    yaml.dump(experiment_parameters, f)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
# Set seed for reproducibility
def set_seeds(seed_value=1):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)  # Set seed for NumPy
    random.seed(seed_value)  # Set seed for Python's Random
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds(exparam.seed)

# Load the dataset
from flex.datasets import load
train_dataset, tests_dataset, val_dataset = load(exparam.data_name, out_dir="../data_original/", training_size=exparam.training_size
                                                 , validation_size=exparam.validation_size, test_size=None)
# Configure the federated data distribution
from flex.data import FedDataDistribution, FedDatasetConfig
config = FedDatasetConfig(
    n_nodes=exparam.n_nodes,
    shuffle=True,
    replacement=False,
    seed=exparam.seed
)
federated_data = FedDataDistribution.from_config(train_dataset, config)


# Assign flip ratios to nodes based on given client flip ratios and label flip ratios
from flex.data.data_poisning import flip_labels, assign_flip_ratios
nodes_label_flip_ratio, label_flip_ratio_nodes = assign_flip_ratios(federated_data.data.keys(),
                                                                    exparam.client_flip_ratio, exparam.label_flip_ratio)

# Set initial bid for each client
if price_test:
    # set bid according to the flip ratio
    Bid_dict = {}
    for node_id, flip_ratio in nodes_label_flip_ratio.items():
        if flip_ratio == 0:
            Bid_dict[node_id] = 14
        elif flip_ratio == 0.4:
            Bid_dict[node_id] = 6
        elif flip_ratio == 0.3:
            Bid_dict[node_id] = 8
        elif flip_ratio == 0.2:
            Bid_dict[node_id] = 10
        elif flip_ratio == 0.1:
            Bid_dict[node_id] = 12
else:
    # set bid with Gaussian random numbers with a mean of 10 and a variance of 1 as initial bid for each node
    Bid_dict = {node_id: np.random.normal(exparam.Bid_mean, exparam.Bid_stddev) for node_id in
                federated_data.data.keys()}

Budget = exparam.Budget
labels_set = set(train_dataset.y_data)
federated_data = federated_data.apply(flip_labels,
                                      nodes_label_flip_ratio=nodes_label_flip_ratio,
                                      label_flip_ratio_nodes=label_flip_ratio_nodes,
                                      labels_set=labels_set,
                                      transfer_node_ids=True)
server_id = "server"
federated_data[server_id] = tests_dataset

import importlib
import inspect
import os
module = importlib.import_module('flex.model')
if hasattr(module, exparam.model_name):
    ModelNet = getattr(module, exparam.model_name)
    class_code = inspect.getsource(ModelNet)
    class_filename = f"{exparam.model_name}_class_definition.py"
    class_filepath = os.path.join(file_path, class_filename)
    with open(class_filepath, 'w') as f:
        f.write(class_code)
    print(f"Class definition for {exparam.model_name} saved to {class_filepath}")
else:
    raise ValueError(f"No class named {exparam.model_name} found in {'flex.model'}")

from flex.pool import init_server_model
from flex.model import FlexModel
@init_server_model
def build_server_model():
    server_flex_model = FlexModel()
    set_seeds(exparam.seed)
    server_flex_model["model"] = ModelNet()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    if exparam.optimizer == "Adam":
        server_flex_model["optimizer_func"] = torch.optim.Adam
        server_flex_model["optimizer_kwargs"] = {'lr': exparam.initial_lr}
    else:
        server_flex_model["optimizer_func"] = torch.optim.SGD
        server_flex_model["optimizer_kwargs"] = {"lr": exparam.initial_lr, "momentum": 0.9}

    return server_flex_model

from flex.data import Dataset
from torch.utils.data import DataLoader

def exponential_lr(round, total_rounds, initial_lr, final_lr):
    return initial_lr * (final_lr / initial_lr) ** (round / (total_rounds-1))
def train(client_flex_model: FlexModel, client_data: Dataset, **kwargs):
    rounds = kwargs.get("round", 0)
    client_dataloader = DataLoader(client_data, batch_size=exparam.batch_size, shuffle=True)
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    if exparam.optimizer == "sgd":
        new_lr = exponential_lr(rounds, exparam.total_rounds, exparam.initial_lr, exparam.final_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    for _ in range(exparam.eachclient_epoch):
        for imgs, labels in client_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
def evaluate_global_model(server_flex_model: FlexModel, test_data: Dataset):
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    # get test data as a torchvision object
    test_dataloader = DataLoader(
        test_data, batch_size=256, shuffle=False, pin_memory=True
    )
    losses = []
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc

from flex.pool import deploy_server_model_pt
from flex.pool.utilsSVselection import random_select_clients_within_budget
from flex.pool import FlexPool
import matplotlib.pyplot as plt
from flex.pool import collect_clients_weights_pt,fed_avg,set_aggregated_weights_pt, set_tensorly_backend,weighted_fed_avg_f
with open(done_result_file, 'r') as f:
    done_result = yaml.unsafe_load(f)


# ==============================================================================
# 帮助函数 (Helper Functions) - 请将这部分放在 train_n_rounds 函数之前
# ==============================================================================
def get_model_weights_vector(model_state_dict):
    """将模型的状态字典展平为一个单一的 NumPy 向量"""
    weights = [param.cpu().numpy().flatten() for param in model_state_dict.values()]
    return np.concatenate(weights)


def cosine_similarity(vec1, vec2):
    """计算两个 NumPy 向量之间的余弦相似度"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0  # 避免除以零
    return np.dot(vec1, vec2) / (norm1 * norm2)


# ==============================================================================
# train_n_rounds 函数 (最终、完整、修正版)
# ==============================================================================
def train_n_rounds(n_rounds, clients_per_round, initial_model_path=None):
    # --- 1. 初始化 ---
    initial_server_model = load_flex_model(initial_model_path)

    @init_server_model
    def build_from_saved():
        return deepcopy(initial_server_model)

    pool_CBE3 = FlexPool.client_server_pool(fed_dataset=federated_data, server_id=server_id, init_func=build_from_saved)
    cbe3_selection_weights = {node_id: 1.0 for node_id in federated_data.data.keys() if node_id != server_id}
    acc_history_CBE3_list = []
    select_list_CBE3 = []

    # 初始化用于存储历史模型权重的变量
    global_weights_t_minus_1 = get_model_weights_vector(pool_CBE3.servers._models[server_id]['model'].state_dict())
    global_weights_t_minus_2 = None

    # 使用测试集评估初始模型性能
    metrics = pool_CBE3.servers.map(evaluate_global_model)
    acc_history_CBE3_list.append(metrics[0][1])

    # --- 超参数设置 (完全遵循原文) ---
    k_approx = 10
    cbe3_eta = np.sqrt(np.log(exparam.n_nodes) / (n_rounds * k_approx))
    # 引入 gamma 探索因子，这是 Exp3.P 的核心，也是数值稳定性的关键
    cbe3_gamma = 0.1

    sm = SeedManager(42)
    sm.set_global_seed()
    sm.register_algorithm("CBE3_Final")

    # --- 2. 训练主循环 ---
    for i in range(n_rounds):
        sm.prepare_round("CBE3_Final", i)
        print(f"\n--- Running Final CBE3 for round: {i + 1} ---")

        # --- 2.1 客户端选择 (使用 Exp3.P 概率模型) ---
        all_client_ids = list(cbe3_selection_weights.keys())
        total_weight = sum(cbe3_selection_weights.values())
        if total_weight <= 1e-9: total_weight = 1.0

        # 【核心修正】使用 Exp3.P 概率公式，确保每个客户端都有保底概率
        selection_probs_this_round = {
            cid: (1 - cbe3_gamma) * (w / total_weight) + (cbe3_gamma / exparam.n_nodes)
            for cid, w in cbe3_selection_weights.items()
        }

        # --- 带预算的概率性选择 ---
        cbe3_selected_ids = []
        current_cost = 0
        candidate_ids = all_client_ids.copy()

        while current_cost < Budget and len(candidate_ids) > 0:
            candidate_probs = {cid: selection_probs_this_round[cid] for cid in candidate_ids}
            current_total_prob = sum(candidate_probs.values())

            # 这个安全卫士现在基本不会被触发，但保留也无妨
            if current_total_prob <= 1e-9:
                normalized_probs_values = [1.0 / len(candidate_ids)] * len(candidate_ids)
            else:
                normalized_probs_values = [p / current_total_prob for p in candidate_probs.values()]

            chosen_client_id = np.random.choice(candidate_ids, p=normalized_probs_values)

            client_bid = Bid_dict.get(chosen_client_id, float('inf'))
            if current_cost + client_bid <= Budget:
                cbe3_selected_ids.append(chosen_client_id)
                current_cost += client_bid

            candidate_ids.remove(chosen_client_id)

        # 回退机制
        if not cbe3_selected_ids and all_client_ids:
            highest_prob_client = max(cbe3_selection_weights, key=cbe3_selection_weights.get)
            if Bid_dict.get(highest_prob_client, float('inf')) <= Budget:
                cbe3_selected_ids = [highest_prob_client]

        select_list_CBE3.append(cbe3_selected_ids)

        # --- 2.2 客户端训练 ---
        if not cbe3_selected_ids:
            print("No clients selected, skipping round.")
            acc_history_CBE3_list.append(acc_history_CBE3_list[-1])
            global_weights_t_minus_2 = global_weights_t_minus_1  # 即使跳过，也要更新历史
            continue

        cbe3_selected_pool = pool_CBE3.clients.select(cbe3_selected_ids)
        pool_CBE3.servers.map(deploy_server_model_pt, cbe3_selected_pool)
        cbe3_selected_pool.map(train, round=i)

        # --- 2.3 贡献度计算 (余弦相似度) 与权重更新 ---
        client_contributions = {}

        if global_weights_t_minus_2 is not None:
            global_update_vector = global_weights_t_minus_1 - global_weights_t_minus_2
        else:
            global_update_vector = None

        if global_update_vector is not None and np.linalg.norm(global_update_vector) > 1e-9:
            for client_id in cbe3_selected_ids:
                client_model_state = cbe3_selected_pool._models[client_id]['model'].state_dict()
                client_weights_vector = get_model_weights_vector(client_model_state)
                client_update_vector = client_weights_vector - global_weights_t_minus_1
                contribution = cosine_similarity(client_update_vector, global_update_vector)
                client_contributions[client_id] = (contribution + 1) / 2.0
        else:  # 第一轮没有全局更新向量，给一个中性奖励
            for client_id in cbe3_selected_ids:
                client_contributions[client_id] = 0.5

                # 权重更新
        for cid in all_client_ids:
            if cid in cbe3_selected_ids:
                prob = selection_probs_this_round[cid]  # 直接使用之前计算好的Exp3.P概率

                reward_i_t = client_contributions[cid]
                unbiased_reward = reward_i_t / prob  # 使用原文的无偏估计

                # 指数更新，现在是数值稳定的
                cbe3_selection_weights[cid] *= np.exp(cbe3_eta * unbiased_reward)

        # --- 2.4 正式聚合与评估 ---
        pool_CBE3.aggregators.map(collect_clients_weights_pt, cbe3_selected_pool)
        pool_CBE3.aggregators.map(fed_avg)
        pool_CBE3.aggregators.map(set_aggregated_weights_pt, pool_CBE3.servers)

        # --- 更新历史权重用于下一轮计算 ---
        global_weights_t_minus_2 = global_weights_t_minus_1
        global_weights_t_minus_1 = get_model_weights_vector(pool_CBE3.servers._models[server_id]['model'].state_dict())

        final_metrics = pool_CBE3.servers.map(evaluate_global_model)
        acc = final_metrics[0][1]
        acc_history_CBE3_list.append(acc)
        print(f'selected clients: {cbe3_selected_ids}')
        print(f"Final CBE3 Accuracy: {acc:.4f}")

        # ... (日志和画图逻辑) ...
        # 注意所有变量名都已修正为CBE3版本
        if (i + 1) % 20 == 0:
            data = {
                'Bid_dict': Bid_dict,
                'nodes_label_flip_ratio': nodes_label_flip_ratio,
                'experiment_parameters': experiment_parameters,
                'acc_history_list': acc_history_CBE3_list,
                'selected_client_id_list': select_list_CBE3,
                'cbe3_weights': cbe3_selection_weights  # 可选：保存最终的权重用于分析
            }
            x_axis = range(len(acc_history_CBE3_list))
            plt.figure(figsize=(10, 6))

            plt.plot(x_axis, done_result['acc_history_SBRO_list'][:i + 2], label='SBRO-FL')
            plt.plot(x_axis, done_result['acc_history_RS_list'][:i + 2], label='RS-FL')
            plt.plot(x_axis, done_result['acc_history_All_list'][:i + 2], label='All-FL')
            plt.plot(x_axis, done_result['acc_history_HQRS_list'][:i + 2], label='HQRS-FL')
            plt.plot(x_axis, acc_history_CBE3_list, label='CBE3-FL')
            plt.title('Federated Learning Accuracy per Round')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.legend()

            round_path = os.path.join(file_path, f"{i + 1}.png")
            plt.savefig(round_path)
            plt.close()

            json_path = os.path.join(file_path, "result_final.json")
            with open(json_path, "w") as f:
                yaml.dump(data, f)

train_n_rounds(exparam.n_rounds, clients_per_round=select_nodes, initial_model_path=INITIAL_MODEL_PATH)


