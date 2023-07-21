import numpy as np
from torch.utils.data import Subset
import math


def split_noniid(train_idcs, train_labels, alpha, n_clients, seed=123):
    
    np.random.seed(seed)
    
    n_classes = 10
    min_size = 0
    min_require_size = 1

    total_data_amount = len(train_idcs)
    #net_dataidx_map = []

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        for y in range(n_classes):
            idx_y = np.argwhere(np.array(train_labels)[np.array(train_idcs)] == y).flatten().tolist()
            # print(f'class {y}\'s amount in client: {len(idx_y)}')
            np.random.shuffle(idx_y)

            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            # total data/client 수 초과된 client는 데이터 할당 X
            proportions = np.array([p * (len(idx_j) < total_data_amount / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
                                              
            # print(f'class {y}\'s distribution')
            
            proportions = (np.cumsum(proportions) * len(idx_y)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_y, proportions))]
            class_distribution = [len(idcs) for idcs in idx_batch]
            
            # print(class_distribution)

            min_size = min([len(idx_j) for idx_j in idx_batch])

    # for i in range(n_clients):
    #     np.random.shuffle(idx_batch[i])
        
    net_dataidx_map = [train_idcs[np.array(idcs)] for idcs in idx_batch] 
    
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


def generate_server_idcs(test_idcs, test_labels, target_class_data_count):
    
    n_class = 10
    server_idcs = []
    remaining_indices = []

    class_idcs = [np.argwhere(np.array(test_labels)[test_idcs] == y).flatten().tolist() for y in range(n_class)]
    
    for class_num, class_index in enumerate(class_idcs):
        if len(class_index) >= target_class_data_count:
            server_idcs.extend(test_idcs[class_index[:target_class_data_count]])
        else:
            server_idcs.extend(test_idcs[class_index])
            remaining_indices.extend(test_idcs[class_index[target_class_data_count:]])

    # If not all classes had enough samples, fill up server_idcs with remaining indices
    while len(server_idcs) < target_class_data_count * n_class and remaining_indices:
        server_idcs.append(remaining_indices.pop())

    # Convert to numpy array
    server_idcs = np.array(server_idcs)

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
