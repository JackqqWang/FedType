import math

import numpy as np
import random
from collections import defaultdict
import torch
torch.manual_seed(10)
np.random.seed(10)

def iid(tokenized_train_set, num_users):

    num_samples = len(tokenized_train_set)
    # Number of samples per user
    samples_per_user = num_samples // num_users

    # Create a list of sample indices and shuffle them
    indices = list(range(num_samples))
    random.shuffle(indices)

    user_groups = {}

    for i in range(num_users):
        start_idx = i * samples_per_user
        end_idx = (i + 1) * samples_per_user

        user_groups[i] = indices[start_idx:end_idx]

    return user_groups


def sst2_noniid(tokenized_train_set, num_users):
    # Separating the indices of positive and negative samples
    positive_indices = [i for i, label in enumerate(tokenized_train_set['label']) if label == 1]
    negative_indices = [i for i, label in enumerate(tokenized_train_set['label']) if label == 0]

    # Shuffle the indices
    random.shuffle(positive_indices)
    random.shuffle(negative_indices)

    # Calculate the number of samples to select
    num_pos = int(math.ceil(0.7 * len(positive_indices)))
    num_neg = int(math.ceil(0.3 * len(negative_indices)))

    # Calculate the number of samples per user
    pos_per_client = num_pos // num_users
    neg_per_client = num_neg // num_users

    user_groups = {}
    for i in range(num_users):
        start_pos = i * pos_per_client
        end_pos = min((i + 1) * pos_per_client, num_pos) if i != num_users - 1 else num_pos
        start_neg = i * neg_per_client
        end_neg = min((i + 1) * neg_per_client, num_neg) if i != num_users - 1 else num_neg

        user_groups[i] = positive_indices[start_pos:end_pos] + negative_indices[start_neg:end_neg]

    return user_groups


def ag_news_noniid(tokenized_train_set, num_users):

    num_shards, num_items = 200, 300  # Adjust num_items based on your dataset size
    idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([]) for i in range(num_users)}
    user_groups = {i: [] for i in range(num_users)}

    # Assuming the labels are in a 'labels' column in the dataset
    idxs_labels = [(i, label) for i, label in enumerate(tokenized_train_set['label'])]
    idxs_labels = sorted(idxs_labels, key=lambda x: x[1])  # sort based on labels
    idxs = [i for i, _ in idxs_labels]


    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:

            user_groups[i] += list(idxs[rand * num_items: (rand + 1) * num_items])

    return user_groups





def fmnist_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])

    return dict_users

def dirichlet_noniid(dataset, num_users, alpha=0.5):
    min_size = 0
    K = len(dataset.classes)
    N = len(dataset)
    y_train = np.array(dataset.targets)
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

def fmnist_noniid(dataset, num_users):

    num_shards = num_users * 2
    # case, 10 clients, 50000 training data, 2500 per shard, 
    # 0,..., 0, 1, ..., 1, 2, ..., 2, ..., 9, ..., 9
    #   5000  ,    5000   ,   5000   , ..., 5000
    #
    num_imgs = len(dataset)//num_shards
    idx_shard = [i for i in range(num_shards)]
    user_groups = {i: [] for i in range(num_users)}
    idxs_labels = [(i, label) for i, label in enumerate(dataset.targets)]
    idxs_labels = sorted(idxs_labels, key=lambda x: x[1])  # sort based on labels
    idxs = [i for i, _ in idxs_labels]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            user_groups[i] += list(idxs[rand * num_imgs: (rand + 1) * num_imgs])

    return user_groups


def dirichlet_fmnist_noniid(dataset, num_users, alpha=0.5):
    min_size = 0
    K = len(dataset.classes)
    N = len(dataset)
    y_train = np.array(dataset.targets)
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map



def cifar100_iid(dataset, num_users):

    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])

    return dict_users

def dirichlet_cifar100_noniid(dataset, num_users, alpha=0.5):
    min_size = 0
    K = len(dataset.classes)
    N = len(dataset)
    y_train = np.array(dataset.targets)
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def cifar100_noniid(dataset, num_users):

    num_shards, num_imgs = num_users * 20, len(dataset)//num_shards
    idx_shard = [i for i in range(num_shards)]
    user_groups = {i: [] for i in range(num_users)}

    idxs_labels = [(i, label) for i, label in enumerate(dataset.targets)]
    idxs_labels = sorted(idxs_labels, key=lambda x: x[1])  # sort based on labels
    idxs = [i for i, _ in idxs_labels]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 20, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # dict_users[i] = np.concatenate(
            #     (dict_users[i], idxs[rand*num_items:(rand+1)*num_items]), axis=0)
            user_groups[i] += list(idxs[rand * num_imgs: (rand + 1) * num_imgs])

    return user_groups






def cifar10_iid(dataset, num_users):

    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users

# new non_iid
def dirichlet_cifar10_noniid(dataset, num_users, alpha=0.5):
    min_size = 0
    K = len(dataset.classes)
    N = len(dataset)
    y_train = np.array(dataset.targets)
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map



def cifar10_noniid(dataset, num_users):

    num_shards = num_users * 2
    # case, 10 clients, 50000 training data, 2500 per shard, 
    # 0,..., 0, 1, ..., 1, 2, ..., 2, ..., 9, ..., 9
    #   5000  ,    5000   ,   5000   , ..., 5000
    #
    num_imgs = len(dataset)//num_shards
    idx_shard = [i for i in range(num_shards)]
    user_groups = {i: [] for i in range(num_users)}
    idxs_labels = [(i, label) for i, label in enumerate(dataset.targets)]
    idxs_labels = sorted(idxs_labels, key=lambda x: x[1])  # sort based on labels
    idxs = [i for i, _ in idxs_labels]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            user_groups[i] += list(idxs[rand * num_imgs: (rand + 1) * num_imgs])

    return user_groups