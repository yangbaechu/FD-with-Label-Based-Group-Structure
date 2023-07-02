import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

def train_op(model, loader, optimizer, epochs=1, grad_clip=None):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Check if loss is valid
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                raise ValueError("Loss is NaN or Infinity. Check your model and training parameters.")
                
            running_loss += loss.detach().item() * y.shape[0]
            samples += y.shape[0]

            loss.backward()

            # Optionally apply gradient clipping
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

    # Switch back to evaluation mode
    model.eval()
    # print(f"running_loss: {running_loss}")
    # print(f"samples:{samples}")
    return running_loss / samples


def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return correct / samples

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, T=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, student_outputs, labels, teacher_outputs):
        hard_loss = F.cross_entropy(student_outputs, labels) * (1 - self.alpha)
        soft_loss = self.alpha * F.kl_div(
            F.log_softmax(student_outputs / self.T, dim=1),
            F.softmax(teacher_outputs / self.T, dim=1),
            reduction="batchmean",
        )
        return hard_loss + soft_loss
    

def eval_for_server(model, loader):
    model.eval()
    samples, correct = 0, 0
    from collections import defaultdict

    label_correct = defaultdict(int)  # Counts of correct predictions for each label
    label_samples = defaultdict(int)  # Counts of total samples for each label
    label_predicted = defaultdict(int)  # Counts of predicted labels
    label_soft_sum = defaultdict(float)  # Sum of soft outputs per label

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            soft_output = F.softmax(y_, dim=1)
            _, predicted = torch.max(y_.data, 1)

            for label in y.tolist():
                label_samples[label] += 1

            for label in predicted.tolist():
                label_predicted[label] += 1

            for i, label in enumerate(y.tolist()):
                label_soft_sum[label] += soft_output[i][label].item()

            correct_preds = predicted == y
            for label, correct in zip(y.tolist(), correct_preds.tolist()):
                if correct:
                    label_correct[label] += 1

    label_accuracies = {
        label: label_correct.get(label, 0) / label_samples[label]
        for label in label_samples
    }

    label_diff = {
        label: label_predicted[label] - label_samples[label] for label in label_samples
    }

    return label_accuracies, label_predicted, label_soft_sum, label_diff


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def get_dW(target, minuend, subtrahend):
    for name in target:
        target[name].data += minuend[name].data.clone() - subtrahend[name].data.clone()


def reduce_add_average(targets, sources):
    for target in targets:
        for name in target:
            tmp = torch.mean(
                torch.stack([source[name].data for source in sources]), dim=0
            ).clone()
            target[name].data += tmp


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.sum(s1 * s2) / (
                torch.norm(s1) * torch.norm(s2) + 1e-12
            )

    return angles.numpy()


class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data):
        self.model = model_fn().to(device)
        self.data = data
        self.W = {key: value for key, value in self.model.named_parameters()}

    def evaluate(self, loader=None):
        return eval_op(self.model, self.eval_loader if not loader else loader)


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, T=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, student_outputs, labels, teacher_outputs):
        hard_loss = F.cross_entropy(student_outputs, labels) * (1 - self.alpha)
        soft_loss = self.alpha * F.kl_div(
            F.log_softmax(student_outputs / self.T, dim=1),
            F.softmax(teacher_outputs / self.T, dim=1),
            reduction="batchmean",
        )
        return hard_loss + soft_loss


class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, idnum, distill_data=None, batch_size=128, train_frac=0.7):
        super().__init__(model_fn, data)
        self.optimizer = optimizer_fn(self.model.parameters())

        self.data = data
        n_train = int(len(data) * train_frac)
        n_eval = len(data) - n_train

        data_train, data_eval = torch.utils.data.random_split(self.data, [n_train, n_eval])

        self.train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)

        if distill_data:
            self.distill_data = TensorDataset(*distill_data)  # assuming distill_data is a tuple (inputs, teacher_probs)
            self.distill_loader = DataLoader(self.distill_data, batch_size=batch_size, shuffle=True)
            
        else:
            self.distill_data = None
            self.distill_loader = None

        self.id = idnum

        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}

        self.loss_fn = DistillationLoss()


    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)
        self.W_original = self.W

    def compute_weight_update(self, epochs=1, distill_epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)

        # Distillation training
        if self.distill_loader is not None:
            for ep in range(distill_epochs):
                running_loss, samples = 0.0, 0
                for x, y, teacher_y in self.distill_loader:
                    x, y, teacher_y = x.to(device), y.to(device), teacher_y.to(device)

                    self.optimizer.zero_grad()

                    outputs = self.model(x)
                    loss = self.loss_fn(outputs, y, teacher_y)

                    running_loss += loss.detach().item() * y.shape[0]
                    samples += y.shape[0]

                    loss.backward()

                    self.optimizer.step()

        # Regular training
        self.optimizer.param_groups[0]["lr"] *= 0.99
        train_stats = train_op(
            self.model,
            self.train_loader if not loader else loader,
            self.optimizer,
            epochs,
        )
        get_dW(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats


    def reset(self):
        copy(target=self.W, source=self.W_original)



class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data): 
        super().__init__(model_fn, data)
        self.model = model_fn().to(device)
        self.loader = DataLoader(self.data, batch_size=128, shuffle=False, pin_memory=True)

        self.model_cache = []
        self.optimizer = optimizer_fn(self.model.parameters())

    # method to generate distillation data
    def make_distillation_data(self):
        self.model.train()  # set the model to training mode
        criterion = torch.nn.CrossEntropyLoss()  # define the loss function

        for epoch in range(40):  # run for 40 epochs
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()  # clear the gradients
                outputs = self.model(inputs)  # forward pass
                loss = criterion(outputs, labels)  # compute loss
                loss.backward()  # backward pass
                self.optimizer.step()  # update the weights

        # use teacher model to make predictions
        self.model.eval()  # set teacher model to eval mode
        all_outputs = []
        all_inputs = []
        all_labels = []
        with torch.no_grad():
            for data, labels in self.loader:
                data, labels = data.to(device), labels.to(device)
                output = self.model(data)
                all_outputs.append(output)
                all_inputs.append(data)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        # apply softmax to convert to probabilities
        teacher_probs = torch.softmax(all_outputs, dim=1)

        distill_data = (all_inputs, all_labels, teacher_probs) # prepare distill_data as a tuple
        return distill_data  # return distill_data instead of individual arrays




    def evaluate(self, model):
        (
            label_accuracies,
            label_predicted,
            label_soft_sum,
            label_diff,
        ) = eval_for_server(model, self.loader)

        return (
            label_accuracies,
            label_predicted,
            label_soft_sum,
            label_diff,
        )

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients) * frac))

    def aggregate_weight_updates(self, clients):
        reduce_add_average(target=self.W, sources=[client.dW for client in clients])

    def compute_pairwise_similarities(self, clients):
        return pairwise_angles([client.dW for client in clients])

    def cluster_clients(self, S):
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="complete"
        ).fit(-S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten()
        c2 = np.argwhere(clustering.labels_ == 1).flatten()
        return c1, c2

    def cluster_clients_DBSCAN(self, S):
        clustering = DBSCAN(eps=1.5, min_samples=1).fit(-S)  # eps 0.8 ~ 1로 test
        return clustering.labels_

    def cluster_clients_GMM(self, S):
        gmm = GaussianMixture(n_components=3)
        gmm.fit(S)
        labels = np.argmax(gmm.predict_proba(S), axis=1)
        c1 = np.argwhere(labels == 0).flatten()
        c2 = np.argwhere(labels == 1).flatten()
        c3 = np.argwhere(labels == 1).flatten()
        return c1, c2, c3

    def cluster_clients_BGM(self, S):
        bgm = BayesianGaussianMixture(n_components=2)
        bgm.fit(S)
        labels = np.argmax(bgm.predict_proba(S), axis=1)
        c1 = np.argwhere(labels.labels_ == 0).flatten()
        c2 = np.argwhere(labels.labels_ == 1).flatten()
        return c1, c2

    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            reduce_add_average(
                targets=[client.W for client in cluster],
                sources=[client.dW for client in cluster],
            )
        return

    def get_average_dw(self, clients):
        # Initialize an empty dictionary to store the average dW
        avg_dW = {}

        # Iterate through each tensor name in the first client's dW to add up dW from all clients
        for name in clients[0].dW:
            avg_dW[name] = sum(client.dW[name].data for client in clients)

        # Divide the summed dW by the number of clients to compute the average
        for name in avg_dW:
            avg_dW[name] /= len(clients)

        return avg_dW

    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        return torch.norm(
            torch.mean(torch.stack([flatten(client.dW) for client in cluster]), dim=0)
        ).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [
            (
                idcs,
                {name: params[name].data.clone() for name in params},
                [accuracies[i] for i in idcs],
            )
        ]
