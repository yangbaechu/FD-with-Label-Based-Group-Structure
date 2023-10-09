import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.optim.lr_scheduler import StepLR

from models import ConvNet, Representation, Ten_class_classifier, Four_class_classifier


device = "cuda" if torch.cuda.is_available() else "cpu"
step_size = 10  # This means decay the LR every 10 epochs
gamma = 0.9


def cutout_and_rotate(image):
    image = image.clone().detach() # 얕은 복사 문제 주의(원본 유지)
    x_start = np.random.randint(20) # cut out 시작할 x축 위치(0~19 중 1개)
    y_start = np.random.randint(20) # cut out 시작할 y축 위치(0~19 중 1개)

    image[..., x_start:x_start+9, y_start:y_start+9] = 255 / 2 # 해당 부분 회색 마킹
    return torch.rot90(image, 1, [-2, -1]) # 마지막 두 axis 기준 90도 회전

def cutout(image):
    image = image.clone().detach()  
    x_start = np.random.randint(20) 
    y_start = np.random.randint(20) 
    image[..., x_start:x_start+9, y_start:y_start+9] = 255 / 2
    return image

def rotate(image):
    return torch.rot90(image, 1, [-2, -1])

def color_jitter(image):
    # This is a simple example. In practice, you would adjust brightness, contrast, saturation, etc.
    image = image + torch.randn_like(image) * 0.1
    image = torch.clamp(image, 0, 255)
    return image

def random_flip(image):
    if np.random.rand() > 0.5:
        return torch.flip(image, [-1])
    return image

AUGMENTATIONS = [cutout, rotate, color_jitter, random_flip]

def apply_random_augmentations(image, num=2):
    aug_image = image.clone().detach()
    chosen_augs = np.random.choice(AUGMENTATIONS, size=num, replace=False)
    for aug in chosen_augs:
        aug_image = aug(aug_image)
    return aug_image


def train_op(model, loader, optimizer, epochs=1, grad_clip=None):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    
    running_loss, samples = 0.0, 0
    for x, y in loader:
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if x.size(0) > 1:
            outputs = model(x)
            r = random.random()
            # if r < 1/500:
            #     print('output in train')
            #     print(torch.max(outputs.data, 1)[:10])
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
        else:
            print("Batch size is 1, skipping this batch")
            
    return running_loss / samples


def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if x.size(0) > 1:
                y_ = model(x)
                _, predicted = torch.max(y_.data, 1)

                samples += y.shape[0]
                correct += (predicted == y).sum().item()
            else:
                print('pass this sample for evluation')
    return correct / samples

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits


def copy(target, source):
    target_copy = deepcopy(source)
    target.update(target_copy)

def get_dW(target, subtrahend):
    difference = {}
    for name in target:
        difference[name] = target[name].data.clone() - subtrahend[name].data.clone()
    return difference



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

    return angles.detach().numpy()

class MajorClassFilterDataset(Dataset):
    def __init__(self, dataset, major_classes):
        self.dataset = dataset
        self.major_classes = major_classes
        
        # Filter indices that have labels in major_classes
        self.indices = [i for i in range(len(dataset)) if dataset[i][1] in self.major_classes]

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

class ModifiedLabelsDataset(Dataset):
    def __init__(self, subset, major_class):
        self.subset = subset
        self.major_class = major_class
        self.new_labels = {cls: idx+1 for idx, cls in enumerate(self.major_class)}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, label = self.subset[idx]
        return data, self.new_labels.get(label, 0) # Remap labels of major classes, set others to 0
    
    
class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    # loss 분모 부분의 negative sample 간의 내적 합만을 가져오기 위한 마스킹 행렬
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # loss 분자 부분의 원본 - augmentation 이미지 간의 내적 합을 가져오기 위한 부분
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)


        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

    
class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data):
        self.model = model_fn()
        
#         self.model.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

#         self.model.classifier[3] = torch.nn.Linear(in_features=self.model.classifier[3].in_features, out_features=10)
        # self.model.num_classes = 10
        # self.model.fc = nn.Linear(self.model.fc.in_features, 10) # Resnet 
        # self.model.classifier[1] = torch.nn.Linear(self.model.classifier[3].in_features, 10) #MobileNet
        self.model = self.model.to(device)
        self.data = data
        self.W = {key: value for key, value in self.model.named_parameters()}

    def evaluate(self, loader=None):
        return eval_op(self.classifier, self.eval_loader if not loader else loader)


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, T=0.1):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, student_outputs, teacher_outputs, labels=None):
        # hard_loss = F.cross_entropy(student_outputs, labels) * (1 - self.alpha)
        # print(f'hard loss: {hard_loss}')
        soft_loss = self.alpha * F.kl_div(
            F.log_softmax(student_outputs / self.T, dim=1),
            F.softmax(teacher_outputs / self.T, dim=1),
            reduction="batchmean",
        )
        # print(f'soft loss: {soft_loss}')
        return soft_loss

class ClusterDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, T=0.1):
        super(ClusterDistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, student_outputs, cluster_logits, global_logits, cluster_weight=None, labels=None):
        # hard_loss = F.cross_entropy(student_outputs, labels) * (1 - self.alpha)
        # print(f'hard loss: {hard_loss}')
        if cluster_weight:
            cluster_loss = cluster_weight * self.alpha * F.kl_div(
                F.log_softmax(student_outputs / self.T, dim=1),
                F.softmax(cluster_logits / self.T, dim=1),
                reduction="batchmean",
            )
        else:
            cluster_loss = self.alpha * F.kl_div(
                F.log_softmax(student_outputs / self.T, dim=1),
                F.softmax(cluster_logits / self.T, dim=1),
                reduction="batchmean",
            )
        if cluster_weight:
            global_loss = (1- cluster_weight)*self.alpha * F.kl_div(
            F.log_softmax(student_outputs / self.T, dim=1),
            F.softmax(global_logits / self.T, dim=1),
            reduction="batchmean", 
        )
        else:
            global_loss = self.alpha * F.kl_div(
                F.log_softmax(student_outputs / self.T, dim=1),
                F.softmax(global_logits / self.T, dim=1),
                reduction="batchmean",
            )
        # print(f'soft loss: {soft_loss}')
        return cluster_loss + global_loss

class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, data, idnum, batch_size=128, train_frac=0.7):
        super().__init__(model_fn, data)
        self.classifier = Ten_class_classifier(self.model).to(device)
        self.data = data
        # self.major_class = major_class

        # Extract the features and labels from the data
        indices = list(range(len(data)))
        labels = [label for _, label in data]

        # Split the indices into training and evaluation sets, maintaining the same distribution of labels
        train_indices, eval_indices, _, _ = train_test_split(indices, labels, train_size=train_frac, stratify=labels)
        
        # Create subsets using the split indices
        # print(f'train_daat_len: {len(train_indices)}')
        # print(f'train_daat_len: {len(eval_indices)}')
        data_train = Subset(data, train_indices)
        data_eval = Subset(data, eval_indices)
        self.data_train = data_train
       
        
        self.train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)
        
#         modified_data_train = ModifiedLabelsDataset(data_train, self.major_class)
#         self.new_train_loader = DataLoader(modified_data_train, batch_size=batch_size, shuffle=True)
        
#         modified_data_eval= ModifiedLabelsDataset(data_train, self.major_class)
#         self.new_eval_loader = DataLoader(modified_data_eval, batch_size=batch_size, shuffle=True)
        
        self.id = idnum

        self.dW = {key: torch.zeros_like(value) for key, value in self.classifier.named_parameters()}
        self.W_old = {key: torch.zeros_like(value) for key, value in self.classifier.named_parameters()}

        self.loss_fn = ClusterDistillationLoss()

    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)
        self.W_original = self.W
    
    # train_data 증강을 통해 Representaion Learning
    def learn_representation(self, train_data):

        batch_size = 64
        
        # Extract X_train from train_data
        X_train = [data[0] for data in train_data]
        
        # Determine maximum number of samples that's a multiple of batch_size
        max_samples = (len(X_train) // batch_size) * batch_size
        X_train = X_train[:max_samples]
        
        X_train = torch.stack(X_train)
        X_train = X_train.to(device)
        
        X_train_aug = cutout_and_rotate(X_train) # 각 X_train 데이터에 대하여 augmentation
        X_train_aug = X_train_aug.to(device) # 학습을 위하여 GPU에 선언
        dataset = TensorDataset(X_train, X_train_aug) # augmentation된 데이터와 pair
        
        dataloader = DataLoader(dataset, batch_size = batch_size)
        
        loss_func = SimCLR_Loss(batch_size, temperature = 0.5) # loss 함수 선언
        
        epochs = 10
        
        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # if self.id % 10 == 0:
        #     print(f'Representation Learning on Client {self.id}')

        for i in range(1, epochs + 1):
            total_loss = 0
            for data in dataloader:
                origin_vec = self.model(data[0])
                aug_vec = self.model(data[1])

                loss = loss_func(origin_vec, aug_vec)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # if self.id % 30 == 0:
            #     print('Epoch : %d, Avg Loss : %.4f'%(i, total_loss / len(dataloader)))
            
        return
    # Client Data를 두 클래스로 분류하는 것 학습
    def train_binary_classifier(self, lr):
        self.binary_classifier = Two_class_classifier(self.model).to(device)

        # 1. prepare dataset
        images = []
        targets = []

        for image, target in self.data:
            if target in self.major_class:
                images.append(image)
                targets.append(0)
            else:
                images.append(image)
                targets.append(1)

        # Convert to PyTorch tensors
        images = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.long)

        # Create a TensorDataset with the selected instances and modified labels
        subset = TensorDataset(images, targets)
        class_dataloader = DataLoader(subset, batch_size=64, shuffle=True)

        # 2. Train
        loss_fn = torch.nn.BCEWithLogitsLoss()
        epochs = 30
        self.binary_classifier.train()
        optimizer = torch.optim.Adam(self.binary_classifier.parameters(), lr=lr)

        num_labels = 2  # Since it's binary classification

        for i in range(1, epochs + 1):
            TP = 0  # True Positives
            TN = 0  # True Negatives
            FP = 0  # False Positives
            FN = 0  # False Negatives

            for data, labels in class_dataloader:
                data, labels = data.to(device), labels.to(device)

                logits = self.binary_classifier(data).squeeze()

                # Compute the Binary Cross-Entropy loss
                loss = loss_fn(logits, labels.float())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Convert logits to probabilities using sigmoid and then threshold at 0.5 for predictions
                predictions = (torch.sigmoid(logits) > 0.5).long()

                # Update counts for TP, TN, FP, FN
                TP += torch.sum((predictions == 1) & (labels == 1)).item()
                TN += torch.sum((predictions == 0) & (labels == 0)).item()
                FP += torch.sum((predictions == 1) & (labels == 0)).item()
                FN += torch.sum((predictions == 0) & (labels == 1)).item()

            # Calculate Precision, Recall, and F1-Score
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if i % 5 == 0 and self.id % 10 == 0:
                print(f'Epoch : {i}, Precision : {precision:.2f}, Recall : {recall:.2f}, F1-Score : {f1_score:.2f}')


    def train_classifier(self, lr):        
        copy(target=self.W_old, source=self.W)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        classifier_loss = nn.CrossEntropyLoss()
        epochs = 30
        self.classifier.train()

        # Early stopping parameters
        patience = 3
        best_loss = float('inf')
        epochs_without_improvement = 0

        for i in range(1, epochs + 1):
            correct_train = 0
            total_loss = 0.0
            for data, labels in self.train_loader:
                data, labels = data.to(device), labels.to(device)

                logits = self.classifier(data)
                loss = classifier_loss(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Compute training accuracy
                _, predicted = torch.max(logits, 1)
                correct_train += (predicted == labels).sum().item()
                total_loss += loss.item()

            train_acc = 100 * correct_train / len(self.train_loader.dataset)

            # Evaluation on eval_loader
            self.classifier.eval()  # Set the model to evaluation mode
            correct_eval = 0
            eval_loss = 0.0
            with torch.no_grad():
                for data, labels in self.eval_loader:
                    data, labels = data.to(device), labels.to(device)
                    logits = self.classifier(data)
                    loss = classifier_loss(logits, labels)
                    eval_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    correct_eval += (predicted == labels).sum().item()

            eval_acc = 100 * correct_eval / len(self.eval_loader.dataset)
            avg_eval_loss = eval_loss / len(self.eval_loader)

            # Early stopping check
            if avg_eval_loss < best_loss:
                best_loss = avg_eval_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement == patience:
                # print("Early stopping triggered.")
                break

            # if self.id % 100 == 5 and i % 5 == 0:
            #     print(f'Epoch {i} - Train Loss: {total_loss / len(self.train_loader):.4f}, Train Accuracy: {train_acc:.2f}%, Eval Accuracy: {eval_acc:.2f}%')
                
        self.dW = get_dW(target=self.W, subtrahend=self.W_old)
        
        return


    def train_four_class_classifier(self, lr):        
        self.optimizer = torch.optim.Adam(self.four_class_classifier.parameters(), lr=lr)
        classifier_loss = nn.CrossEntropyLoss()
        epochs = 50
        self.four_class_classifier.train()

        for i in range(1, epochs + 1):
            for data, labels in self.new_train_loader:
                data, labels = data.to(device), labels.to(device)

                logits = self.four_class_classifier(data)
                loss = classifier_loss(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Evaluation on eval_loader
            self.four_class_classifier.eval()  # Set the model to evaluation mode

            correct_eval_class0 = 0
            total_class0 = 0
            correct_eval_other_classes = 0
            total_other_classes = 0

            with torch.no_grad():
                for data, labels in self.new_eval_loader:
                    data, labels = data.to(device), labels.to(device)
                    logits = self.four_class_classifier(data)
                    _, predicted = torch.max(logits, 1)

                    # Counters for class 0
                    mask_class0 = (labels == 0)
                    correct_eval_class0 += (predicted[mask_class0] == labels[mask_class0]).sum().item()
                    total_class0 += mask_class0.sum().item()

                    # Counters for other classes
                    mask_other_classes = (labels != 0)
                    correct_eval_other_classes += (predicted[mask_other_classes] 
                                                   == labels[mask_other_classes]).sum().item()
                    total_other_classes += mask_other_classes.sum().item()

            if total_class0 > 0:
                eval_acc_class0 = 100 * correct_eval_class0 / total_class0
            else:
                eval_acc_class0 = 0

            if total_other_classes > 0:
                eval_acc_other_classes = 100 * correct_eval_other_classes / total_other_classes
            else:
                eval_acc_other_classes = 0

            if i % 50 == 0:
                print(f'Epoch {i} - Eval Accuracy for Minor: {eval_acc_class0:.2f}%, '
                      f'Eval Acc for Major: {eval_acc_other_classes:.2f}%')

        return

    
#     def dual_distill(self, distill_data, epochs=40, max_grad_norm=1.0):
#         print(f"Type of distill_data: {type(distill_data)}")
#         for i, item in enumerate(distill_data):
#             print(f"Type of distill_data[{i}] is {type(item)}")
        
#         self.distill_loader = DataLoader(TensorDataset(*distill_data), batch_size=128, shuffle=True)
#         copy(target=self.W_old, source=self.W)


#         # Distillation training
#         if self.distill_loader is not None:
#             for ep in range(epochs):
#                 running_loss, samples = 0.0, 0
#                 for x, cluster_logit, global_logit in self.distill_loader:
#                     x, cluster_logit, global_logit = x.to(device), cluster_logit.to(device), global_logit.to(device)

#                     self.optimizer.zero_grad()

#                     outputs = self.model(x)
#                     loss = self.loss_fn(outputs, cluster_logit, global_logit)
#                     now_loss = loss.detach().item() * x.shape[0]

#                     running_loss += now_loss
#                     # if ep % 10 == 0:
#                     #     print(f'loss: {now_loss}')
#                     samples += x.shape[0]

#                     loss.backward()
                    
#                     if max_grad_norm is not None:
#                         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

#                     self.optimizer.step()

#         get_dW(target=self.dW, minuend=self.W, subtrahend=self.W_old)
#         return
    
    def dual_distill(self, distill_loader, silhouette, epochs=50):
        self.classifier.train()
        self.optimizer = torch.optim.Adam(self.classifier.parameters())
        self.loss_fn = ClusterDistillationLoss()  # Make sure this loss function can handle dual logits

        for ep in range(1, epochs + 1):
            running_loss, samples, correct_predictions = 0.0, 0, 0

            for i, (x, cluster_teacher_y, global_teacher_y) in enumerate(distill_loader):
                self.optimizer.zero_grad()

                x = x.to(device)
                cluster_teacher_y = cluster_teacher_y.to(device)
                global_teacher_y = global_teacher_y.to(device)

                outputs = self.classifier(x)
                _, predicted = torch.max(outputs, 1)

                # Count the number of correct predictions
                # Here, you might need to decide how to compare predictions to teacher labels
                correct_predictions += (predicted == global_teacher_y.argmax(1)).sum().item()

                # Calculate the dual distillation loss here
                # You might want to update your loss function to take two types of teacher logits
                weight = max(0, (silhouette-0.5)*2.5)
                weight = min(weight, 1)
                loss = self.loss_fn(outputs, cluster_teacher_y, global_teacher_y, weight)
                now_loss = loss.detach().item() * x.shape[0]

                running_loss += now_loss
                samples += x.shape[0]

                loss.backward()
                self.optimizer.step()

            # Optionally, print logs
            # if ep % 5 == 0:
            #     average_loss = running_loss / samples
            #     accuracy = (correct_predictions / samples) * 100  # Accuracy as a percentage
            #     print(f'dual_distill epoch {ep}, averaged loss: {average_loss:.4f}, accuracy: {accuracy:.2f}%')

        return

    def distill(self, distill_loader, epochs=50):
        # self.classifier = Ten_class_classifier(self.model).to(device)
        self.classifier.train()
        self.optimizer = torch.optim.Adam(self.classifier.parameters())#, lr=2e-4)
        self.loss_fn = DistillationLoss()

        # copy(target=self.W_old, source=self.W)
        torch.autograd.set_detect_anomaly(True)

        # for g in self.optimizer.param_groups:
        #     g['lr'] = 0.0005
        # Distillation training

        for ep in range(1, epochs + 1):
            running_loss, samples, correct_predictions = 0.0, 0, 0

            for i, (x, teacher_y) in enumerate(distill_loader):
                self.optimizer.zero_grad()

                x, teacher_y = x.to(device), teacher_y.to(device)

                outputs = self.classifier(x)
                _, predicted = torch.max(outputs, 1)

                # Count the number of correct predictions
                correct_predictions += (predicted == teacher_y.argmax(1)).sum().item()

                loss = self.loss_fn(outputs, teacher_y)
                now_loss = loss.detach().item() * x.shape[0]

                running_loss += now_loss
                samples += x.shape[0]

                loss.backward()

                # if max_grad_norm is not None:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()

            # if ep % 5 == 0 and self.id % 5 == 0:
            #     average_loss = running_loss / samples
            #     accuracy = (correct_predictions / samples) * 100  # Accuracy as a percentage
            #     print(f'distill epoch {ep}, averaged loss: {average_loss:.4f}, accuracy: {accuracy:.2f}%')

        # get_dW(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return


    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)

        train_stats = train_op(
            self.model,
            self.train_loader, #if not loader else loader,
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
        self.classifier = Ten_class_classifier(self.model).to(device)
        self.loader = DataLoader(self.data, batch_size=128, shuffle=False, pin_memory=True)
        
        self.model_cache = []
        self.optimizer = optimizer_fn(self.model.parameters())
    
    def create_distill_loader(self, dataset, server_idcs, global_logits, batch_size=64):
        transform = transforms.ToTensor()

        # Extract data samples
        data_samples = [transform(dataset[i][0]) for i in server_idcs]

        # Convert list of tensors to a single tensor
        data_samples = torch.stack(data_samples)

        # Check if all logits in the rows of global_logits are -1
        valid_indices = (global_logits != -1).any(dim=1)

        # Filter out invalid samples
        filtered_samples = data_samples[valid_indices]
        filtered_logits = global_logits[valid_indices]

        # print(f'data sample length: {len(filtered_samples)}')
        # print(f'Valid global logits length: {len(filtered_logits)}')
        # print(f'server_idcs length: {len(server_idcs)}')

        # Create the dataset
        distill_dataset = TensorDataset(filtered_samples, filtered_logits)

        distill_loader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True)

        return distill_loader
    
    
    def create_dual_distill_loader(self, dataset, server_idcs, cluster_logits, global_logits, batch_size=64):
        transform = transforms.ToTensor()

        # Extract data samples
        data_samples = [transform(dataset[i][0]) for i in server_idcs]

        # Convert list of tensors to a single tensor
        data_samples = torch.stack(data_samples)

        # Check if all logits in the rows of global_logits and cluster_logits are -1
        valid_global_indices = (global_logits != -1).any(dim=1)
        valid_cluster_indices = (cluster_logits != -1).any(dim=1)

        # Combine the valid indices for both global and cluster logits
        valid_indices = valid_global_indices & valid_cluster_indices

        # Filter out invalid samples
        filtered_samples = data_samples[valid_indices]
        filtered_global_logits = global_logits[valid_indices]
        filtered_cluster_logits = cluster_logits[valid_indices]

        # Create the dataset
        # filtered_cluster_logits = torch.Tensor(filtered_cluster_logits)

        distill_dataset = TensorDataset(filtered_samples, filtered_cluster_logits, filtered_global_logits)

        distill_loader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True)

        return distill_loader

    
        
    def get_four_class_classifier_probabilities(self, four_class_classifier, major_class):
        device = next(four_class_classifier.parameters()).device
        four_class_classifier.eval()

        correct_predictions = 0
        total_major_class_samples = 0

        all_adjusted_probabilities = []

        for data, labels in self.loader:
            data, labels = data.to(device), labels.to(device)

            outputs = four_class_classifier(data)
            probabilities = F.softmax(outputs, dim=1)
            # print(f'probabilities[0]: {probabilities[0]}')
            _, predicted = torch.max(outputs, 1)

            # Adjust values based on the logic you provided
            for idx, probs in enumerate(probabilities):
                adjusted_probs = [0 for _ in range(10)]

                for i, major_cls_index in enumerate(major_class):
                    adjusted_probs[major_cls_index] = probs[i + 1].item()
                # print(f'adjusted_probs[0]: {adjusted_probs}')

                all_adjusted_probabilities.append(torch.tensor(adjusted_probs))

                if labels[idx].item() in major_class:
                    total_major_class_samples += 1
                    if major_class[predicted[idx].item() - 1] == labels[idx].item():
                        correct_predictions += 1

        accuracy = (correct_predictions / total_major_class_samples) * 100 if total_major_class_samples > 0 else 0
        print("Accuracy of Adjusted Probabilities (Only Major Classes): {:.2f}%".format(accuracy))

        # Convert list of tensors to a 2D tensor
        all_adjusted_probabilities_tensor = torch.stack(all_adjusted_probabilities)
        # print(all_adjusted_probabilities_tensor[:10])
        return all_adjusted_probabilities_tensor, accuracy

    
    def get_cluster_logits(self, client_logits, number_of_cluster, t):
        # Convert the logits to predicted labels by getting the class with the maximum logit for each client
        predicted_labels = [torch.argmax(logits, dim=1) for logits in client_logits]

        # Count occurrences of each predicted label for each client
        num_classes = client_logits[0].shape[1]
        label_counts = torch.zeros((len(client_logits), num_classes))
        for i, labels in enumerate(predicted_labels):
            for label in labels:
                label_counts[i][label] += 1

        label_predicted = pd.DataFrame(label_counts.numpy())  # Convert to DataFrame for clustering
        
        # print(f'label_predicted: {label_predicted}')
        # Cluster based on the label counts
        cluster_idcs = self.cluster_clients_Hierarchical(label_predicted, number_of_cluster, t)

        # Compute the average logits for each cluster
        cluster_logits = []
        for cluster in cluster_idcs:
            cluster_client_logits = [client_logits[i] for i in cluster]
            avg_cluster_logits = torch.mean(torch.stack(cluster_client_logits), dim=0)
            cluster_logits.append(avg_cluster_logits.detach())

        return cluster_logits, cluster_idcs, label_predicted
    
    def evaluate_clustering(self, data, cluster_distribution, cluster_idcs):
        """
        Evaluates clustering using silhouette score and adjusted rand index (ARI).

        Args:
        - data (array-like): Original data points.
        - cluster_distribution (list): List of proportions for the real clusters.
        - cluster_idcs (list): Lists of indices for each predicted cluster.

        Returns:
        - tuple: (Silhouette score, ARI score)
        """

        # Create labels based on cluster indices
        predicted_labels = [None] * len(data)
        for cluster_label, cluster in enumerate(cluster_idcs):
            for idx in cluster:
                predicted_labels[idx] = cluster_label

        # Check the number of unique labels
        unique_labels = np.unique([label for label in predicted_labels if label is not None])

        n = len(data)
        print(n)
        start_idx = 0
        true_labels = []
        current_label = 0  # Initialize current label to 0
        for proportion in cluster_distribution:
            end_idx = start_idx + int(n * proportion)
            true_labels.extend([current_label] * (end_idx - start_idx))
            start_idx = end_idx
            current_label += 1  # Increment the label for the next cluster

        # # Compute the ARI score
        # print(f'predicted_labels: {predicted_labels}')
        # print(f'true_labels: {true_labels}')
        ari = adjusted_rand_score(true_labels, predicted_labels)

        # Compute the silhouette score only if there is more than one unique label
        if len(unique_labels) > 1:
            silhouette = silhouette_score(data, predicted_labels)
        else:
            silhouette = np.nan  # Set to nan if only one unique label

        # print(f"Adjusted Rand Index: {ari:.2f}, Silhouette Score: {silhouette:.2f}")

        return silhouette, ari




    def get_clients_logit(self, classifier):
        classifier.eval()  # set classifier to eval mode

        all_outputs = []

        class_correct = {}  # Counter for correct predictions for each class
        class_total = {}    # Counter for total samples of each class

        for i, (data, labels) in enumerate(self.loader):
            data, labels = data.to(device), labels.to(device)

            output = classifier(data)
            class_predictions = output.argmax(dim=1)

            for i in range(len(labels)):
                label_item = labels[i].item()

                # Initialize counters if the class is encountered for the first time
                if label_item not in class_total:
                    class_total[label_item] = 0
                    class_correct[label_item] = 0

                class_total[label_item] += 1

                if class_predictions[i] == label_item:
                    class_correct[label_item] += 1

                probs = torch.softmax(output[i], dim=0)
                all_outputs.append(probs)

        # Convert list of tensors into a single 2D tensor
        all_outputs = torch.stack(all_outputs)

        return all_outputs

    
    def get_clients_logit_simclr(self, binary_classifier, classifier, major_class):
        binary_classifier.eval()  # set binary_classifier to eval mode
        classifier.eval()  # set classifier to eval mode

        all_outputs = []

        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        TN = 0  # True Negatives

        for i, (data, labels) in enumerate(self.loader):
            data, labels = data.to(device), labels.to(device)
            binary_logit = binary_classifier(data).squeeze()
            major_class_predictions = (torch.sigmoid(binary_logit) > 0.5).long()

            for i in range(len(labels)):
                label_item = labels[i].item()

                if label_item in major_class:  # If the actual label is major class
                    if major_class_predictions[i] == 0:  # Correct prediction
                        TP += 1
                    else:  # False negative
                        FN += 1
                else:  # If the actual label is minor class
                    if major_class_predictions[i] == 1:  # Correct prediction
                        TN += 1
                    else:  # False positive
                        FP += 1

                input_data = data[i].unsqueeze(0)

                if major_class_predictions[i] == 0:  # If major class
                    output = classifier(input_data)
                    probs = torch.softmax(output, dim=1)
                    all_outputs.append(probs.squeeze(0))
                else:
                    placeholder = torch.full((1, 10), -1).to(device)
                    all_outputs.append(placeholder.squeeze(0))

        all_outputs = torch.stack(all_outputs)

        # Calculate precision, recall and F1-score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}")

        return all_outputs




    def get_global_logits(self, logits_list):
        # Stack the logits from different models along a new dimension
        stacked_logits = torch.stack(logits_list, dim=0)

        # Check if all logits along the new dimension are -1
        all_minus_one = (stacked_logits == -1).all(dim=0)

        # Compute the mean of the logits across the models
        mean_logits = stacked_logits.mean(dim=0)

        # Normalize the logits using softmax so the sum is 1
        global_logits = F.softmax(mean_logits, dim=-1)

        # Set to -1 where all models output -1
        global_logits[all_minus_one] = -1

        # print(global_logits[:10])

        return global_logits.detach()



    def check_cluster(self, model):
        model.eval()
        label_predicted = defaultdict(int)  # Counts of predicted labels

        with torch.no_grad():
            for i, (x, y) in enumerate(self.loader):
                x, y = x.to(device), y.to(device)
                if x.size(0) > 1:
                    y_ = model(x)
                    r =  random.random()
                    # if r < 1/100:
                    #     print(y_[:4])
                    #     print(y_.data[:4])
                    _, predicted = torch.max(y_.data, 1)
                    # if r < 1/100:
                    #     print(predicted)
                    for label in predicted.tolist():
                        label_predicted[label] += 1
                else:
                    print("x is only one!")
                    # print('predicted label')
                    # print(label_predicted)

        return label_predicted
    
    def evaluate_distil(self, model):
        model.eval()  # Set model to evaluation mode
        samples, correct = 0, 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.loader):
                # Evaluate only on 20% of the data
                if i > len(self.loader) // 5:
                    break

                x, y = x.to(device), y.to(device)

                y_ = model(x)
                # if random.randint(1, 100) == 1:
                #     print(f'y\'s length: {len(y_[0])}')
                #     print(f'y: {y_[0]}')
                    # print(y_[0])
                _, predicted = torch.max(y_.data, 1)
                # if random.randint(1, 100) == 1:
                #     print(f'predicted: {predicted[:10]}')
                #     print(f'y: {y[:10]}')
                samples += y.shape[0]
                correct += (predicted == y).sum().item()
                # print(f'samples: {samples}, correct: {correct}, acc: {correct / samples}')
        return correct / samples if samples > 0 else 0

    
    def select_clients(self, clients, frac=1.0):
        # Filter clients with more than 5 data points
        eligible_clients = [client for client in clients if len(client.data) > 5]

        return random.sample(eligible_clients, int(len(eligible_clients) * frac))


    def aggregate_weight_updates(self, clients):
        reduce_add_average(target=self.W, sources=[client.dW for client in clients])

    def compute_pairwise_similarities(self, clients):
        return pairwise_angles([client.dW for client in clients])

    def cluster_clients(self, S):
        # Fit the hierarchical clustering model
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="complete"
        ).fit(-S)

        number_of_clusters = clustering.n_clusters_

        # Get the cluster indices
        cluster_indices = []
        for cluster in range(number_of_clusters):
            cluster_indices.append(np.argwhere(clustering.labels_ == cluster).flatten())

        return cluster_indices, number_of_clusters

    def cluster_clients_DBSCAN(self, S):
        clustering = DBSCAN(eps=1.5, min_samples=1).fit(-S)  # eps 0.8 ~ 1로 test
        return clustering.labels_

    def cluster_clients_GMM(self, S, number_of_cluster):
        gmm = GaussianMixture(n_components=number_of_cluster)
        gmm.fit(S)
        labels = np.argmax(gmm.predict_proba(S), axis=1)
        
        cluster_idcs = []
        
        for cluster in range(number_of_cluster):
            cluster_idcs.append(np.argwhere(labels == cluster).flatten())
        # c1 = np.argwhere(labels == 0).flatten()
        # c2 = np.argwhere(labels == 1).flatten()
        # c3 = np.argwhere(labels == 2).flatten()
        return cluster_idcs
    
    

    def cluster_clients_KMeans(self, S, number_of_cluster):
        kmeans = KMeans(n_clusters=number_of_cluster,  n_init=10, random_state=0)
        labels = kmeans.fit_predict(S)

        cluster_idcs = []
        for cluster in range(number_of_cluster):
            cluster_idcs.append(np.argwhere(labels == cluster).flatten())
        return cluster_idcs
    
    def cluster_clients_Hierarchical(self, S, number_of_cluster, t):
        # Step 2: Normalize the data using Min-Max scaling
        scaler = MinMaxScaler()
        S_normalized = scaler.fit_transform(S)
        # print('S_normalized')
        # print(S_normalized)
        # Step 1 & 3: Initialize AgglomerativeClustering with distance_threshold=0.5 and n_clusters=None
        agglomerative_clustering = AgglomerativeClustering(distance_threshold=t, n_clusters=None)

        # Fit the model and get labels
        labels = agglomerative_clustering.fit_predict(S_normalized)
        print(labels)
        # Find the unique labels to identify the number of clusters formed
        unique_labels = np.unique(labels)

        # Initialize list to store indices for each cluster
        cluster_idcs = []

        # Populate cluster indices
        for cluster in unique_labels:
            cluster_idcs.append(np.argwhere(labels == cluster).flatten())

        return cluster_idcs

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
