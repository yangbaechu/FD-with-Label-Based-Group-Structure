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
        # print(f'loss: {running_loss}')
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

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits


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

class NTD_Loss(nn.Module):
    """Not-true Distillation Loss"""

    def __init__(self, num_classes=10, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, T=0.1):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, student_outputs, labels, teacher_outputs):
        # hard_loss = F.cross_entropy(student_outputs, labels) * (1 - self.alpha)
        soft_loss = self.alpha * F.kl_div(
            F.log_softmax(student_outputs / self.T, dim=1),
            F.softmax(teacher_outputs / self.T, dim=1),
            reduction="batchmean",
        )
        return soft_losss
    
class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data):
        self.model = model_fn(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.model = self.model.to(device)
        self.data = data
        self.W = {key: value for key, value in self.model.named_parameters()}

    def evaluate(self, loader=None):
        return eval_op(self.model, self.eval_loader if not loader else loader)


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, T=0.1):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, student_outputs, labels, teacher_outputs):
        # hard_loss = F.cross_entropy(student_outputs, labels) * (1 - self.alpha)
        # print(f'hard loss: {hard_loss}')
        soft_loss = self.alpha * F.kl_div(
            F.log_softmax(student_outputs / self.T, dim=1),
            F.softmax(teacher_outputs / self.T, dim=1),
            reduction="batchmean",
        )
        # print(f'soft loss: {soft_loss}')
        return soft_loss


class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, idnum, batch_size=512, train_frac=0.7):
        super().__init__(model_fn, data)
        self.optimizer = optimizer_fn(self.model.parameters())

        self.data = data
        
        n_train = int(len(data) * train_frac)
        n_eval = len(data) - n_train
        
        data_train, data_eval = torch.utils.data.random_split(self.data, [n_train, n_eval])
        
        self.train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)
        
        self.id = idnum

        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}

        self.loss_fn = DistillationLoss()


    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)
        self.W_original = self.W

    def distill(self, distill_data, epochs=40, max_grad_norm=1.0):
        self.distill_loader = DataLoader(TensorDataset(*distill_data), batch_size=512, shuffle=True)
        copy(target=self.W_old, source=self.W)

        # Distillation training
        if self.distill_loader is not None:
            for ep in range(epochs):
                running_loss, samples = 0.0, 0
                for x, y, teacher_y in self.distill_loader:
                    x, y, teacher_y = x.to(device), y.to(device), teacher_y.to(device)

                    self.optimizer.zero_grad()

                    outputs = self.model(x)        
                    loss = self.loss_fn(outputs, y, teacher_y)
                    now_loss = loss.detach().item() * y.shape[0]

                    running_loss += now_loss
                    # if ep % 10 == 0:
                    #     print(f'loss: {now_loss}')
                    samples += y.shape[0]

                    loss.backward()
                    
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()

        get_dW(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return
        
    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)

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



from collections import Counter

class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data): 
        super().__init__(model_fn, data)
        self.loader = DataLoader(self.data, batch_size=512, shuffle=False, pin_memory=True)

        self.model_cache = []
        self.optimizer = optimizer_fn(self.model.parameters())


   # method to generate distillation data
    def make_distillation_data(self, data_per_class):
        self.model.train()  # set the model to training mode
        criterion = torch.nn.CrossEntropyLoss()  # define the loss function

        for epoch in range(50):  
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
        class_count = {}  # keep count of examples per class

        with torch.no_grad():
            for data, labels in self.loader:
                data, labels = data.to(device), labels.to(device)
                output = self.model(data)

                for i in range(len(labels)):
                    label = labels[i]
                    if label.item() not in class_count:
                        class_count[label.item()] = 0
                    if class_count[label.item()] < data_per_class:
                        all_outputs.append(output[i])
                        all_inputs.append(data[i])
                        all_labels.append(label)
                        class_count[label.item()] += 1

        all_outputs = torch.stack(all_outputs, dim=0)
        all_inputs = torch.stack(all_inputs, dim=0)
        all_labels = torch.stack(all_labels, dim=0)
        # apply softmax to convert to probabilities
        teacher_probs = torch.softmax(all_outputs, dim=1)

        distill_data = (all_inputs, all_labels, teacher_probs) # prepare distill_data as a tuple
        return distill_data  # return distill_data instead of individual arrays

    def get_clients_logit(self, model, data_per_class):
        model.eval()  # set teacher model to eval mode
        all_outputs = []
        all_inputs = []
        all_labels = []
        # class_count = {}  # keep count of examples per class

        with torch.no_grad():
            for data, labels in self.loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)

                for i in range(len(labels)):
                    label = labels[i]
                    all_outputs.append(output[i].unsqueeze(0))  # appending 2D tensor of shape [1, 62]
                    all_inputs.append(data[i].unsqueeze(0))  # appending 2D tensor
                    all_labels.append(label.unsqueeze(0))  # appending 1D tensor

        # Then convert lists of tensors into single tensors
        all_outputs = torch.cat(all_outputs, dim=0)  # you can use torch.cat again since all tensors are 2D now
        all_inputs = torch.cat(all_inputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)


        # apply softmax to convert to probabilities
        teacher_probs = torch.softmax(all_outputs, dim=1)
        
        # print(all_labels[0])
        # print(teacher_probs[0])
        
        distill_data = (all_inputs, all_labels, teacher_probs) # prepare distill_data as a tuple
        return distill_data  # return distill_data instead of individual arrays

    def get_global_logits(self, client_logits):
        avg_logits = torch.mean(torch.stack(client_logits), dim=0)
        return avg_logits

    
    # predicted, soft_sum: Unlabel data도 가능
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
    
    def evaluate_distil(self, model):
        model.eval()  # Set model to evaluation mode
        samples, correct = 0, 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.loader):
                # Evaluate only on 20% of the data
                if i > len(self.loader) // 20:
                    break

                x, y = x.to(device), y.to(device)

                y_ = model(x)
                _, predicted = torch.max(y_.data, 1)

                samples += y.shape[0]
                correct += (predicted == y).sum().item()

        return correct / samples if samples > 0 else 0

    
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
