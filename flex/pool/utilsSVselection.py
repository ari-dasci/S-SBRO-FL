import math
import pulp as pp
import torch
import numpy as np
import itertools

from pkg_resources import non_empty_lines

from flex.pool import collect_clients_weights_pt
from flex.pool import fed_avg
from torch.utils.data import DataLoader
from math import factorial
import copy
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque
def debug_plot(y_dict, name, highlight_ids=[]):
    ids = range(len(y_dict))
    y = [y_dict[i] for i in range(len(y_dict))]

    plt.figure(figsize=(10, 6))

    # 绘制默认颜色的点
    for i in ids:
        if i in highlight_ids:
            plt.scatter(i, y[i], color='red', label='highlight' if i == highlight_ids[0] else "")
        else:
            plt.scatter(i, y[i], color='blue', label='no sorted' if i == 0 else "")

    plt.title(name)
    plt.xlabel('id')
    plt.ylabel(name)
    plt.legend()
    plt.show()

def prospect_theory(Repuation_dict,γ=2.5, α=0.3, β=0.9):
    values = [v[-1] for v in Repuation_dict.values()]
    R0 = sum(values) / len(values)
    result = {}
    for key, R_list in Repuation_dict.items():
        R = R_list[-1]
        z = R - R0
        if z >= 0:
            result[key] = z ** α
        else:
            result[key] = -γ * ((-z) ** β)
    # debug_plot(result,'prospect_theory')
    # _ = result
    min_value = min(result.values())
    result = {key: value - min_value for key, value in result.items()}
    # debug_plot(_,'softmax')

    return result

def exact_shapley_value(clients, baseline_performance, val_data, device,*args, **kwargs):
    """
    Calculate exact Shapley values for each client in federated learning.

    Parameters:
    -----------
    clients : FlexPool
        FlexPool containing client models.
    baseline_performance : float
        The performance of the global model without any client.
    val_data : Dataset
        The validation dataset to evaluate the model.
    device : str
        The device to use for the model.

    Returns:
    --------
    shapley_values : dict
        Dictionary of Shapley values for each client.
    """
    n = len(clients._models)
    shapley_values = {client_id: 0 for client_id in clients._models}
    cache = {}
    ignore_weights = kwargs.get("ignore_weights", None)
    if ignore_weights is None:
        ignore_weights = ["num_batches_tracked"]
    def get_subset_value(subset):
        """
        Get the performance value of a subset of clients.

        Parameters:
        -----------
        subset : list
            List of client IDs representing the subset.

        Returns:
        --------
        float
            The performance value of the subset.
        """
        subset_key = tuple(sorted(subset))
        if subset_key not in cache:
            if len(subset) == 0:
                subset_acc = baseline_performance
            else:
                subset_params = [collect_clients_weights_pt.__wrapped__(clients._models[client_id]) for client_id in subset]
                aggregated_params = fed_avg.__wrapped__(subset_params)
                model = copy.deepcopy(clients._models[subset[0]]['model'])
                weight_dict = model.state_dict()
                for layer_key, new in zip(weight_dict, aggregated_params):
                    try:
                        if len(new) != 0:  # Do not copy empty layers
                            weight_dict[layer_key].copy_(new)
                    except TypeError:  # new has no len property
                        weight_dict[layer_key].copy_(new)

                model.eval()
                test_acc = 0
                total_count = 0
                model = model.to(device)
                # get test data as a torchvision object
                test_dataloader = DataLoader(
                    val_data, batch_size=128, shuffle=False, pin_memory=True
                )
                losses = []
                with torch.no_grad():
                    for data, target in test_dataloader:
                        total_count += target.size(0)
                        data, target = data.to(device), target.to(device)
                        output = model(data) 
                        pred = output.data.max(1, keepdim=True)[1]
                        test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                test_acc /= total_count
                subset_acc = test_acc
            cache[subset_key] = subset_acc
        return cache[subset_key]

    for client in clients._models:
        for k in range(n):
            subsets = itertools.combinations([c for c in clients._models if c != client], k)
            for subset in subsets:
                subset = list(subset)
                subset_with_client = subset + [client]

                subset_acc_with_client = get_subset_value(subset_with_client)
                subset_acc_without_client = get_subset_value(subset)

                marginal_contribution = subset_acc_with_client - subset_acc_without_client
                shapley_values[client] += (marginal_contribution * factorial(k) * factorial(n - k - 1)) / factorial(n)

    return shapley_values


def calculate_loo_contribution(clients, baseline_performance, val_data, device, *args, **kwargs):
    """
    Calculates each client's marginal contribution using the Leave-One-Out (LOO) method.

    This is a simplified contribution metric for ablation studies, comparing the performance
    of the full coalition of selected clients with the performance of the coalition
    missing just one client.

    Parameters:
    -----------
    clients : FlexPool
        FlexPool containing the selected client models for the current round.
    baseline_performance : float
        The performance of the global model from the previous round (before this round's updates).
    val_data : Dataset
        The validation dataset to evaluate the model performance.
    device : str
        The device to use for model evaluation (e.g., "cuda" or "cpu").

    Returns:
    --------
    loo_contributions : dict
        A dictionary where keys are client_ids and values are their LOO marginal contribution.
    """
    selected_ids = clients.actor_ids
    if not selected_ids:
        return {}

    loo_contributions = {client_id: 0 for client_id in selected_ids}
    cache = {}

    def get_subset_value(subset_ids):
        """
        Internal helper function to get the performance value of a subset of clients.
        It uses a cache to avoid re-evaluating the same subset.
        """
        subset_key = tuple(sorted(subset_ids))
        if subset_key in cache:
            return cache[subset_key]

        if not subset_ids:
            # If the subset is empty, performance is the baseline from the previous round
            cache[subset_key] = baseline_performance
            return baseline_performance

        # Aggregate the models of the clients in the subset
        subset_params = [collect_clients_weights_pt.__wrapped__(clients._models[client_id]) for client_id in subset_ids]
        aggregated_params = fed_avg.__wrapped__(subset_params)

        # Create a clean model instance for evaluation
        model = copy.deepcopy(clients._models[subset_ids[0]]['model'])
        weight_dict = model.state_dict()
        for layer_key, new_weights in zip(weight_dict, aggregated_params):
            try:
                if len(new_weights) != 0:
                    weight_dict[layer_key].copy_(new_weights)
            except TypeError:
                weight_dict[layer_key].copy_(new_weights)

        # Evaluate the aggregated model
        model.eval()
        model = model.to(device)
        test_dataloader = DataLoader(val_data, batch_size=256, shuffle=False, pin_memory=True)

        test_acc = 0
        total_count = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                total_count += target.size(0)
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

        accuracy = test_acc / total_count if total_count > 0 else 0.0
        cache[subset_key] = accuracy
        return accuracy

    # 1. Calculate the performance of the grand coalition (all selected clients)
    v_S = get_subset_value(selected_ids)

    # 2. Iterate through each client to calculate their marginal contribution
    for client_id in selected_ids:
        # Create the coalition without the current client
        subset_minus_client = [cid for cid in selected_ids if cid != client_id]

        # Calculate the performance of the coalition without the client
        v_S_minus_i = get_subset_value(subset_minus_client)

        # The LOO contribution is the performance drop when the client is removed
        marginal_contribution = v_S - v_S_minus_i
        loo_contributions[client_id] = marginal_contribution

    return loo_contributions


def calculate_tmc_shapley(clients, baseline_performance, val_data, device, *args, **kwargs):
    """
    使用截断蒙特卡洛（TMC）方法近似计算每个客户端的Shapley值。

    该方法通过对客户端排列进行随机抽样来估算边际贡献，
    从而避免了精确计算的指数级复杂度。

    Args:
        clients (FlexPool): 包含当前轮次被选中客户端模型的池。
        baseline_performance (float): 上一轮的全局模型性能（基线）。
        val_data (Dataset): 用于评估模型性能的验证数据集。
        device (str): "cuda" 或 "cpu"。
        *args, **kwargs: 可选参数，其中 'mc_iterations' 用于指定蒙特卡洛迭代次数。

    Returns:
        dict: 一个字典，键是客户端ID，值是其近似的Shapley值。
    """
    selected_ids = clients.actor_ids
    if not selected_ids:
        return {}

    # 从kwargs获取迭代次数，如果没有则设置一个合理的默认值
    mc_iterations = kwargs.get('mc_iterations', 20 * len(selected_ids))

    tmc_shapley_values = {client_id: 0.0 for client_id in selected_ids}
    cache = {}  # 用于缓存已计算过的子集性能，提高效率

    def get_subset_performance(subset_ids):
        """内部辅助函数，用于评估给定客户端子集的模型性能。"""
        subset_key = tuple(sorted(subset_ids))
        if subset_key in cache:
            return cache[subset_key]

        if not subset_ids:
            return baseline_performance

        # 聚合模型
        subset_params = [collect_clients_weights_pt.__wrapped__(clients._models[cid]) for cid in subset_ids]
        aggregated_params = fed_avg.__wrapped__(subset_params)

        model = copy.deepcopy(clients._models[selected_ids[0]]['model'])
        weight_dict = model.state_dict()
        for layer_key, new_weights in zip(weight_dict, aggregated_params):
            try:
                if len(new_weights) != 0: weight_dict[layer_key].copy_(new_weights)
            except TypeError:
                weight_dict[layer_key].copy_(new_weights)

        # 评估模型
        model.eval()
        model = model.to(device)
        test_dataloader = DataLoader(val_data, batch_size=256, shuffle=False, pin_memory=True)

        test_acc = 0
        total_count = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                total_count += target.size(0)
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

        accuracy = test_acc / total_count if total_count > 0 else 0.0
        cache[subset_key] = accuracy
        return accuracy

    # TMC-Shapley 主循环
    for _ in range(mc_iterations):
        # 随机生成一个客户端排列
        shuffled_ids = random.sample(selected_ids, len(selected_ids))

        preceding_coalition_perf = baseline_performance

        for i in range(len(shuffled_ids)):
            client_id = shuffled_ids[i]

            # 包含当前客户端的联盟
            current_coalition_ids = shuffled_ids[:i + 1]
            current_coalition_perf = get_subset_performance(current_coalition_ids)

            # 计算边际贡献
            marginal_contribution = current_coalition_perf - preceding_coalition_perf

            # 累加贡献值以计算平均值
            tmc_shapley_values[client_id] += marginal_contribution

            # 更新前序联盟的性能
            preceding_coalition_perf = current_coalition_perf

    # 计算Shapley值的平均值
    for client_id in tmc_shapley_values:
        tmc_shapley_values[client_id] /= mc_iterations

    return tmc_shapley_values


def update_reputation(Repuation_dict, Bid_dict, shapley_values,sv_history, ω, ψ, window, penalty_growth_rate=1.5):
    """
    Update reputation of nodes based on Shapley values, bid prices, and performance history.

    Parameters:
    -----------
    Repuation_dict : dict
        Dictionary of reputation values for each node.
    Bid_dict : dict
        Dictionary of bid values for each node.
    shapley_values : dict
        Dictionary of Shapley values for each node.
    poorperformanceCount : dict
        Dictionary of poor performance count for each node.
    ω : float
        Reward coefficient for positive contributions.
    ψ : float
        Punishment coefficient for negative contributions (should be negative).
    round: int
        Current round number.

    """
    sum_positive_sv = sum(sv for sv in shapley_values.values() if sv > 0)
    sum_positive_bid = sum(Bid_dict[node_id] for node_id, sv in shapley_values.items() if sv > 0)
    for node_id, sv in shapley_values.items():
        if sv <= 0:
            recent_values = sv_history[node_id][-window:]
            # 计算小于0的次数
            recent_errors = sum(1 for v in recent_values if v < 0)
            # Update bad count
            UD = -ψ * (penalty_growth_rate ** recent_errors)
        else:
            relative_positive_contribution = sv / sum_positive_sv if sum_positive_sv != 0 else 0
            relative_bid = Bid_dict[node_id] / sum_positive_bid if sum_positive_bid != 0 else 0
            exp_term = -relative_positive_contribution / relative_bid
            UD = ω * (1 - np.exp(exp_term))
        # Update reputation
        if node_id in Repuation_dict:
            Repuation_dict[node_id].append(Repuation_dict[node_id][-1] + UD)


def select_nodes(Reputation_dict, Bid_dict, Budget, γ=-2.5, α=0.3, β=0.9, round=1, participation_count_5rounds={}):
    #   convert the reputation values to weights using the prospect theory function
    W = prospect_theory(Reputation_dict, γ, α, β)

    #  calculate the weight threshold
    sorted_data = sorted([v for v in W.values()])
    weight_threshold = np.percentile(sorted_data, 50)

    #  adjust the weights based on the number of times a node has been selected in the last five rounds
    for node_id, count in participation_count_5rounds.items():
        #
        decay_factor = 0.5 ** count
        W[node_id] *= decay_factor

    #   create an optimization problem
    m = pp.LpProblem(sense=pp.LpMaximize)
    x = {node_id: pp.LpVariable(f'x{node_id}', cat='Binary') for node_id in Reputation_dict.keys()}

    #   add the objective function
    epsilon = 1e-6  #
    m += pp.lpSum([(W[node_id] + epsilon) * x[node_id] for node_id in Reputation_dict.keys()])

    #  add the budget constraint
    m += pp.lpSum([Bid_dict[node_id] * x[node_id] for node_id in Reputation_dict.keys()]) <= Budget

    #  add the weight threshold constraint
    for node_id in Reputation_dict.keys():
        m += W[node_id] * x[node_id] >= weight_threshold * x[node_id]

    #  solve the problem
    m.solve(pp.PULP_CBC_CMD(msg=False))

    # get the selected node IDs
    selected_nodes = [node_id for node_id in Reputation_dict.keys() if pp.value(x[node_id]) == 1]

    # ----- roll back -----
    if not selected_nodes:
       #  if no nodes are selected, re-run the optimization without the weight threshold constraint
        m2 = pp.LpProblem(sense=pp.LpMaximize)
        m2 += pp.lpSum([(W[node_id] + epsilon) * x[node_id] for node_id in Reputation_dict.keys()])
        m2 += pp.lpSum([Bid_dict[node_id] * x[node_id] for node_id in Reputation_dict.keys()]) <= Budget
        #  remove the weight threshold constraint
        m2.solve(pp.PULP_CBC_CMD(msg=False))
        selected_nodes = [node_id for node_id in Reputation_dict.keys() if pp.value(x[node_id]) == 1]

    return selected_nodes

def select_nodes_linear_nofilter(Reputation_dict, Bid_dict, Budget, participation_count_5rounds={}):
    """
    Robust version with status check and safe fallback.
    """

    # === Calculate weights ===
    W = {node_id: rep_list[-1] for node_id, rep_list in Reputation_dict.items()}

    min_W = min(W.values())
    W = {k: v - min_W for k, v in W.items()}

    # === Create optimization problem ===
    m = pp.LpProblem("ClientSelection_Linear", sense=pp.LpMaximize)
    x = {node_id: pp.LpVariable(f'x_{node_id}', cat='Binary') for node_id in Reputation_dict.keys()}

    epsilon = 1e-6
    # Objective
    m += pp.lpSum([(W[node_id] + epsilon) * x[node_id] for node_id in Reputation_dict.keys()])

    # Budget constraint
    m += pp.lpSum([Bid_dict[node_id] * x[node_id] for node_id in Reputation_dict.keys()]) <= Budget

    # === Solve ===
    m.solve(pp.PULP_CBC_CMD(msg=False))

    # === Check status ===
    if m.status != 1:
        print(f"[WARNING] Solver status not optimal: {pp.LpStatus[m.status]}")
        selected_nodes = []
    else:
        selected_nodes = [node_id for node_id in Reputation_dict.keys() if pp.value(x[node_id]) == 1]

    # === Fallback if no nodes selected ===
    if not selected_nodes:
        print("[INFO] Triggering fallback selection (without hard threshold).")

        # Redefine problem & variables
        m2 = pp.LpProblem("ClientSelection_Linear_Fallback", sense=pp.LpMaximize)
        x2 = {node_id: pp.LpVariable(f'x2_{node_id}', cat='Binary') for node_id in Reputation_dict.keys()}

        # Objective
        m2 += pp.lpSum([(W[node_id] + epsilon) * x2[node_id] for node_id in Reputation_dict.keys()])

        # Budget constraint only
        m2 += pp.lpSum([Bid_dict[node_id] * x2[node_id] for node_id in Reputation_dict.keys()]) <= Budget

        m2.solve(pp.PULP_CBC_CMD(msg=False))

        if m2.status != 1:
            print(f"[WARNING] Fallback solver status not optimal: {pp.LpStatus[m2.status]}")
            selected_nodes = []
        else:
            selected_nodes = [node_id for node_id in Reputation_dict.keys() if pp.value(x2[node_id]) == 1]

    return selected_nodes

def random_select_clients_within_budget(Bid_dict, Budget):
    """
    Select clients based on their bids and a budget constraint.
    Bid_dict : dict
        Dictionary of bid values for each client.
    Budget : int or float
        The budget for client selection.
    Returns:
    --------
    selected_clients : list
        List of selected client IDs.

    """
    selected_clients = []
    total_cost = 0
    client_ids = list(Bid_dict.keys())

    random.shuffle(client_ids)

    for client_id in client_ids:
        bid = Bid_dict[client_id]
        if total_cost + bid <= Budget:
            selected_clients.append(client_id)
            total_cost += bid
        else:
            break

    return selected_clients

def random_select_clients_within_budget_in_normal(Bid_dict, Budget, good_list):
    """
    Select clients based on their bids and a budget constraint.
    Bid_dict : dict
        Dictionary of bid values for each client.
    Budget : int or float
        The budget for client selection.
    Returns:
    --------
    selected_clients : list
        List of selected client IDs.

    """
    selected_clients = []
    total_cost = 0

    random.shuffle(good_list)

    for client_id in good_list:
        bid = Bid_dict[client_id]
        if total_cost + bid <= Budget:
            selected_clients.append(client_id)
            total_cost += bid
        else:
            break

    return selected_clients
