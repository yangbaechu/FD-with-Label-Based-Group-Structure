import numpy as np
from torch.utils.data import Subset
import math

def split_noniid_original(train_idcs, train_labels, alpha, n_clients):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    """
    n_classes = 10
    label_distribution = np.random.dirichlet([alpha] * n_classes, n_clients)

    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten().tolist() for y in range(n_classes)] #MNiST
    #class_idcs = [np.argwhere(np.array(train_labels)[np.array(train_idcs)] == y).flatten().tolist() for y in range(n_classes)] CIFAR

    total_data = [len(idcs) for idcs in class_idcs]
    client_idcs = [[] for _ in range(n_clients)]

   # Determine common amount of data each client should have
    total_data_amount = sum([len(idcs) for idcs in class_idcs])
    client_data_amount = (total_data_amount//10) // len(client_idcs)

    for i, (client, fracs) in enumerate(zip(client_idcs, label_distribution)):

        # Keep track of how much data has been allocated to this client
        allocated_data_amount = 0

        for j, (idcs, frac) in enumerate(zip(class_idcs, fracs)):
            idcs_len = int(total_data[j]*frac*0.1)

            # Adjust the data length if it exceeds the remaining data that this client can receive
            remaining_data_amount = client_data_amount - allocated_data_amount
            if idcs_len > remaining_data_amount:
                idcs_len = remaining_data_amount

            # if idcs_len == 0:
            #     print("no data allocated")
            # if idcs_len > len(idcs):
            #     print(f'allocated data: {idcs_len}, left data: {len(idcs)}')

            client.extend(idcs[:idcs_len])
            allocated_data_amount += idcs_len
            del idcs[:idcs_len]

        # If a client has not received enough data, fill it with the remaining needed data from the available class indices
        if allocated_data_amount < client_data_amount:
            print(f'client {i} didn\'t recieved enough data!')
            print(f'target data: {client_data_amount}, allocated_data_amount: {allocated_data_amount}')
            print(f'label_distribution: {np.round(fracs*100)}')
            for idcs in class_idcs:
                remaining_data_amount = client_data_amount - allocated_data_amount
                if remaining_data_amount <= len(idcs):
                    client.extend(idcs[:remaining_data_amount])
                    del idcs[:remaining_data_amount]
                    break

    # print([len(c_id) for c_id in client_idcs])
    
    client_idcs = [train_idcs[np.array(idcs)] for idcs in client_idcs] 

    
    return client_idcs

def split_noniid(train_idcs, train_labels, alpha, n_clients, seed=123):
    
    np.random.seed(seed)
    
    n_classes = 10
    min_size = 0
    min_require_size = 10

    total_data_amount = len(train_idcs)
    #net_dataidx_map = []

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        for y in range(n_classes):
            idx_y = np.argwhere(train_labels[train_idcs] == y).flatten().tolist()
            np.random.shuffle(idx_y)

            proportions = np.random.dirichlet(np.repeat(alpha, n_clients)
            # total data/client 수 초과된 client는 데이터 할당 X
            proportions = np.array([p * (len(idx_j) < total_data_amount / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            # 합이 1이 되도록 정규화
            proportions = proportions / proportions.sum()
            #각 class가 받는 데이터 숫자 구하기
            proportions = (np.cumsum(proportions) * len(idx_y)).astype(int)[:-1]
            print(f'class {y}\'s distribution: {proportions}')

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_y, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # for i in range(n_clients):
    #     # np.random.shuffle(idx_batch[i])
        
    net_dataidx_map = [train_idcs[np.array(idcs)] for idcs in idx_batch] 
    
    print([len(idcs) for idcs in net_dataidx_map])
    
    return net_dataidx_map



def split_contain_every_class(train_idcs, train_labels, n_clients, data_per_class, Imbalance_ratio):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to deterministic rule
    """

    data_per_class = data_per_class
    n_cluster = 3
    client_per_cluster = 3
    n_class = 20
    Imbalance_ratio = Imbalance_ratio
    client_idcs = [[] for _ in range(n_clients)]

    class_idcs = [
        np.where(train_labels == c)[0].tolist()[:4000] for c in range(n_class)
    ]

    for c in range(n_cluster):
        for client in range(client_per_cluster):
            # split indices for class c across clients for that class
            idx = []
            for class_num, class_index in enumerate(class_idcs):
                start = (n_cluster * 3 + n_clients) * data_per_class
                end = start + data_per_class
                # Allocate 2 times more data of class 0, 1, 2 to cluster 1 and so on
                if class_num // 3 == c:
                    end += int(data_per_class * (Imbalance_ratio - 1))
                idx.extend(class_index[start:end])

            client_idcs[client + c * client_per_cluster] += idx

    # Convert to numpy array
    client_idcs = [np.array(idcs) for idcs in client_idcs]

    return client_idcs


def split_not_contain_every_class(train_idcs, train_labels, n_clients):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to deterministic rule
    """

    n_classes = 3  # we have three classes 0, 1 and 2
    client_idcs = [[] for _ in range(n_clients)]
    clients_per_class = (
        n_clients // n_classes
    )  # assume n_clients is a multiple of n_classes

    for c in range(n_classes):
        # find indices of class c
        class_idcs_1 = np.where(train_labels == c)[0].tolist()[:4000]
        class_idcs_2 = np.where(train_labels == c + 3)[0].tolist()[:4000]

        # split indices for class c across clients for that class
        for i in range(clients_per_class):
            start = i * 200
            end = start + 200

            client_idcs[i * n_classes + c] += list(class_idcs_1[start:end])
            client_idcs[i * n_classes + c] += list(class_idcs_2[start:end])

    # Convert to numpy array
    client_idcs = [np.array(idcs) for idcs in client_idcs]

    return client_idcs


def generate_server_idcs(test_idcs, test_labels, start_idx):
    
    n_class = 10
    server_idcs = []
    data_per_class = 1000

    class_idcs = [np.where(test_labels == c)[0].tolist() for c in range(n_class)]

    for class_num, class_index in enumerate(class_idcs):
        start = 0
        end = start + data_per_class
        server_idcs.extend(class_index[start:end])

    # Convert to numpy array
    server_idcs = np.array(server_idcs)
    server_idcs += start_idx

    return server_idcs



class CustomSubset(Subset):
    """A custom subset class with customizable data transformation"""

    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y
