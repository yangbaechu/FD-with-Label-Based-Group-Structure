import numpy as np
from torch.utils.data import Dataset, Subset
import math


def split_noniid(train_idcs, train_labels, alpha, n_clients, seed=123):
    
    np.random.seed(seed)
    
    n_classes = 10
    min_size = 0
    min_require_size = 1

    total_data_amount = len(train_idcs)
    #net_dataidx_map = []

    while min_size < min_require_size:
        print('not enough data!')
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



def split_2class_plus_alpha(train_idcs, train_labels, n_clients, seed=123):
    np.random.seed(seed)

    n_classes = 10
    clients_per_class = n_clients // (n_classes // 2)
    idx_batch = [[] for _ in range(n_clients)]
    remaining_data = []

    for y in range(n_classes):
        idx_y = np.argwhere(np.array(train_labels)[np.array(train_idcs)] == y).flatten().tolist()
        np.random.shuffle(idx_y)
        
        split_idx = int(0.8 * len(idx_y))
        idx_y_split = np.array_split(idx_y[:split_idx], clients_per_class)
        remaining_data.extend(idx_y[split_idx:])
        
        for i, client_id in enumerate(range((y // 2) * clients_per_class, (y // 2 + 1) * clients_per_class)):
            idx_batch[client_id] += idx_y_split[i].tolist()

    np.random.shuffle(remaining_data)
    remaining_data_split = np.array_split(remaining_data, n_clients)

    for i in range(n_clients):
        idx_batch[i] += remaining_data_split[i].tolist()

    net_dataidx_map = [train_idcs[np.array(idcs)] for idcs in idx_batch] 

    # print class distribution for each client
    # print("Class distribution per client:")
    # for i, idcs in enumerate(idx_batch):
    #     class_counts = {label:0 for label in range(n_classes)}
    #     for idx in idcs:
    #         class_counts[train_labels[train_idcs[idx]]] += 1
    #     print(f"Client {i}: {class_counts}")

    return net_dataidx_map


def split_contain_multiclass(train_idcs, train_labels, n_clients, instances_per_class, instances_per_class_per_client, cluster_distribution, seed=123):
    
    np.random.seed(seed)
    
    n_classes = 10  # You may want to make this a parameter or calculate it based on the data
    
    num_groups = len(instances_per_class_per_client)
    print(f'num_groups: {num_groups}')
    # assert n_clients % num_groups == 0, "Total number of clients must be divisible by the number of groups"
    

    idx_batch = [[] for _ in range(n_clients)]
   
    
    client_id = 0
    print(cluster_distribution)
       
    # Assuming each group in instances_per_class_per_client has the same number of classes
    for i, class_group in enumerate(instances_per_class_per_client):
        for _ in range(int(cluster_distribution[i] * n_clients)):
            for y in class_group:
                idx_y = np.argwhere(np.array(train_labels)[np.array(train_idcs)] == y).flatten().tolist()
                np.random.shuffle(idx_y)
                
                idx_batch[client_id] += idx_y[:instances_per_class]
                
            client_id += 1
    net_dataidx_map = [train_idcs[np.array(idcs)] for idcs in idx_batch]
    
    return net_dataidx_map


def split_3class_unbalanced(train_idcs, train_labels, n_clients, cluster_distribution, seed=123):
    
    np.random.seed(seed)
    
    n_classes = 10
    classes_per_group = 3
    data_per_class = 50

    # Number of clients per group based on the given distribution
    clients_per_group = [int(dist * n_clients) for dist in cluster_distribution]
    idx_batch = [[] for _ in range(n_clients)]
    major_class_per_client = []
    
    for group in range(n_classes // classes_per_group):
        for y in range(group * classes_per_group, (group + 1) * classes_per_group):
            if y >= n_classes:
                continue
            idx_y = np.argwhere(np.array(train_labels)[np.array(train_idcs, dtype=int)] == y).flatten().tolist()
            np.random.shuffle(idx_y)
            idx_y = idx_y[:data_per_class * clients_per_group[group]]  # Only take the first 50 * clients_per_group data
            
            # Split indices of the same class into multiple chunks
            idx_y_split = np.array_split(idx_y, clients_per_group[group])

            # Assign each chunk to a different client
            for i in range(clients_per_group[group]):
                client_id = sum(clients_per_group[:group]) + i
                idx_batch[client_id] += idx_y_split[i].tolist()
        
    for i in range(clients_per_group[group]):
        major_class_per_client.append(assigned_classes)
    net_dataidx_map = [train_idcs[np.array(idcs, dtype=int)] for idcs in idx_batch] 
    
    # # print class distribution for each client
    # print("Class distribution per client:")
    # for i, idcs in enumerate(idx_batch):
    #     class_counts = {label:0 for label in range(n_classes)}
    #     for idx in idcs:
    #         class_counts[train_labels[train_idcs[idx]]] += 1
    #     print(f"Client {i}: {class_counts}")
    
    return net_dataidx_map, major_class_per_client


# version that return major_class information
def split_7plus3class_unbalanced(train_idcs, train_labels, n_clients, cluster_distribution, data_per_class_3, data_per_class_7, seed=123):
    np.random.seed(seed)
    
    n_classes = 10
    classes_per_group = 3
    # data_per_class_3 = 100
    # data_per_class_7 = 14
    print(f'data_per_class_3: {data_per_class_3}, data_per_class_7: {data_per_class_7}')
    # Number of clients per group based on the given distribution
    clients_per_group = [int(dist * n_clients) for dist in cluster_distribution]
    idx_batch = [[] for _ in range(n_clients)]
    major_class_per_client = []

    for group in range(n_classes // classes_per_group):
        assigned_classes = list(range(group * classes_per_group, (group + 1) * classes_per_group))
        remaining_classes = [i for i in range(n_classes) if i not in assigned_classes]

        # Distributing the first 3 classes
        for y in assigned_classes:
            if y >= n_classes:
                continue
            idx_y = np.argwhere(np.array(train_labels)[np.array(train_idcs, dtype=int)] == y).flatten().tolist()
            np.random.shuffle(idx_y)
            idx_y = idx_y[:data_per_class_3 * clients_per_group[group]]  
            
            idx_y_split = np.array_split(idx_y, clients_per_group[group])
            for i in range(clients_per_group[group]):
                client_id = sum(clients_per_group[:group]) + i
                idx_batch[client_id] += idx_y_split[i].tolist()

        # Add major classes to the major_class_per_client list
        for i in range(clients_per_group[group]):
            major_class_per_client.append(assigned_classes)

        # Distributing the remaining 7 classes for this group
        for y in remaining_classes:
            idx_y = np.argwhere(np.array(train_labels)[np.array(train_idcs, dtype=int)] == y).flatten().tolist()
            np.random.shuffle(idx_y)
            idx_y = idx_y[:data_per_class_7 * clients_per_group[group]] 
            
            idx_y_split = np.array_split(idx_y, clients_per_group[group])
            for i in range(clients_per_group[group]):
                client_id = sum(clients_per_group[:group]) + i
                idx_batch[client_id] += idx_y_split[i].tolist()

    net_dataidx_map = [train_idcs[np.array(idcs, dtype=int)] for idcs in idx_batch] 
    
    return net_dataidx_map, major_class_per_client

# def split_not_contain_every_class(train_idcs, train_labels, n_clients):
#     """
#     Splits a list of data indices with corresponding labels
#     into subsets according to deterministic rule
#     """

#     n_classes = 10  
#     client_idcs = [[] for _ in range(n_clients)]
#     client_per_group = n_clients // (n_classes//2)  # clients per group

#     # find indices of each class
#     class_idcs = [np.argwhere(np.array(train_labels)[np.array(train_idcs)] == y).flatten().tolist() for y in range(n_classes)]

#     # loop over groups of clients
#     for group in range(n_classes // 2):
#         # find two classes for this group of clients
#         c1 = group * 2
#         c2 = c1 + 1

#         # split indices for these two classes across this group of clients
#         for i in range(group * client_per_group, (group+1) * client_per_group):
#             idcs_per_client_1 = len(class_idcs[c1]) // client_per_group
#             idcs_per_client_2 = len(class_idcs[c2]) // client_per_group

#             # calculate start and end for each class
#             client_idx = i - group * client_per_group
#             start_1 = client_idx * idcs_per_client_1
#             end_1 = start_1 + idcs_per_client_1
#             start_2 = client_idx * idcs_per_client_2
#             end_2 = start_2 + idcs_per_client_2

#             # assign indices
#             client_idcs[i] += class_idcs[c1][start_1:end_1]
#             client_idcs[i] += class_idcs[c2][start_2:end_2]

#     # Convert to numpy array
#     client_idcs = [np.array(idcs) for idcs in client_idcs]
    
#     # Print class distribution for some clients
#     for i in range(min(5, n_clients)):  # adjust the number of clients to display
#         client_class_idcs = [np.argwhere(np.array(train_labels)[client_idcs[i]] == y).flatten().tolist() for y in range(n_classes)]
#         print(f"Client {i} class distribution: {list(map(len, client_class_idcs))}")


#     return client_idcs



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
    
class CombinedCustomSubset(Dataset):
    """A combined custom subset class for both server and client data"""

    def __init__(self, dataset, server_idcs, client_idcs, subset_transform=None):
        self.dataset = dataset
        self.server_idcs = server_idcs
        self.client_idcs = client_idcs
        self.subset_transform = subset_transform
        self.total_indices = server_idcs + client_idcs

    def __getitem__(self, idx):
        true_idx = self.total_indices[idx]
        x, y = self.dataset[true_idx]

        if self.subset_transform and (true_idx in self.server_idcs):
            x = self.subset_transform(x)

        return x, y

    def __len__(self):
        return len(self.total_indices)
