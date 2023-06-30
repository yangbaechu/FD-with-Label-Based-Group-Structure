import numpy as np
from torch.utils.data import Subset


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    """
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [
        np.argwhere(train_labels[train_idcs] == y).flatten() for y in range(n_classes)
    ]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(
            np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
        ):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


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


def generate_server_idcs(test_idcs, test_labels, n_class=10):
    """
    Generate server indices with identical label distribution as client_idcs
    """

    server_idcs = []
    data_per_class = 100

    class_idcs = [np.where(test_labels == c)[0].tolist() for c in range(n_class)]

    for class_num, class_index in enumerate(class_idcs):
        start = 0
        end = start + data_per_class
        server_idcs.extend(class_index[start:end])

    # Convert to numpy array
    server_idcs = np.array(server_idcs)
    server_idcs += 10000

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
