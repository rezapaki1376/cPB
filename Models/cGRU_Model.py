import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from river import metrics
from utils.utils import (
    customized_loss,
    accuracy,
    cohen_kappa,
    kappa_temporal,
    get_samples_outputs,
    get_pred_from_outputs, kappa_temporal_score,
)
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import copy
from Models.pretrain import GRU_Model  # Import the GRU model
import matplotlib.pyplot as plt
import pickle
class cGRU:
    """
    Continual learning framework using a single GRU model with weight-sharing and continual adaptation.
    """

    def __init__(self, hidden_size=50, device=None, stride=1, lr=0.01,
                 seq_len=10, pretrain_model_addr='', input_size=2,
                 epoch_size=10, batch_first=True, output_size=2, **kwargs):
        """
        Initializes the continual learning framework with a single GRU model.
        """
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.stride = stride
        self.seq_len = seq_len
        self.lr = lr
        self.hidden_size = hidden_size
        self.device = device
        self.pretrain_model_addr = pretrain_model_addr
        self.epoch_size = epoch_size
        self.input_size = input_size
        self.batch_first = batch_first
        self.output_size = output_size

        # Performance tracking dictionary
        self.performance = {}
        self.performance_anytime = {}

        # Initialize the model (Pretrained or from scratch)
        self.model = GRU_Model(hidden_size=self.hidden_size, input_size=self.input_size, output_size = self.output_size)
        if pretrain_model_addr:
            with open(self.pretrain_model_addr, "rb") as fp:
                self.model.load_state_dict(pickle.load(fp), strict=False)

    def _cut_in_sequences(self, x, y):
        """Splits the data into sequences of specified length."""
        seqs_features, seqs_targets = [], []
        for i in range(0, len(x), self.stride):
            if len(x) - i >= self.seq_len:
                seqs_features.append(x[i: i + self.seq_len, :].astype(np.float32))
                if y is not None:
                    seqs_targets.append(np.asarray(y[i: i + self.seq_len], dtype=np.int_))
        return np.asarray(seqs_features), np.asarray(seqs_targets)

    def _convert_to_tensor_dataset(self, x, y=None):
        """Converts the input features and labels into a TensorDataset."""
        x, y = self._cut_in_sequences(x, y)
        x = torch.tensor(x)
        if y is not None:
            y = torch.tensor(y).long()
            return data_utils.TensorDataset(x, y)
        return x

    def _load_batch(self, x, y=None):
        """Prepares the batch for training."""
        batch = self._convert_to_tensor_dataset(x, y)
        batch_loader = DataLoader(batch, batch_size=batch.tensors[0].size()[0], drop_last=False)
        for x, y_seq in batch_loader:
            break
        return x, torch.tensor(y), y_seq

    def learn_many(self, x, y):
        """Trains the model on a batch of data."""
        x = np.array(x)
        y = list(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        x, y, _ = self._load_batch(x, y)

        for _ in range(self.epoch_size):
            optimizer.zero_grad()
            y_pred, _ = self.model(x)
            y_pred = get_samples_outputs(y_pred)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
    def learn_many_anytime(self, x, y):
        """
        Trains the model on a batch of data for a specific task.

        Parameters
        ----------
        x : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Labels corresponding to the input features.
        task_number : int
            Index of the task to train on.
        """
        x = np.array(x)
        y = list(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        x, y, _ = self._load_batch(x, y)        
        for _ in range(self.epoch_size):
            optimizer.zero_grad()
            y_pred,_ = self.model(x)
            loss = self.loss_fn(y_pred[:, -1, :], y[self.seq_len-1:])
            # loss = self.loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

    def predict_many(self, x, y, task_number):
        """Predicts for a batch of data."""
        with torch.no_grad():
            x = np.array(x)
            y = list(y)
            x, y, _ = self._load_batch(x, y)
            y_pred,_ = self.model(x)
            y_pred = get_samples_outputs(y_pred)
            pred, _ = get_pred_from_outputs(y_pred)
            kappa = cohen_kappa(y, pred).item()
            acc = accuracy_score(np.array(y), np.array(pred))

            # Ensure the dictionary is initialized before using it
            if f'task_{task_number}' not in self.performance:
                self.performance[f'task_{task_number}'] = {
                    'acc': [],
                    'kappa': [],
                    'predictions': []
                }
            
            # print(pred)
            # print(acc)
            # print(kappa)
            # print(ssssss)

            self.performance[f'task_{task_number}']['acc'].append(acc)
            self.performance[f'task_{task_number}']['kappa'].append(kappa)
            self.performance[f'task_{task_number}']['predictions'].append(pred)
    def predict_one(self, x, y, task):
        """
        Makes predictions for a batch of data for a specific task.

        Parameters
        ----------
        x : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Labels corresponding to the input features.
        mask_number : int
            Index of the mask to use for predictions.
        task_number : int
            Index of the task to predict on.
        mask_selection : bool, default=False
            Whether to use mask selection.

        Returns
        -------
        None
        """
        with torch.no_grad():
            x = np.array(x)
            y = list(y)
            x, y, _ = self._load_batch(x, y)
            y_pred,_ = self.model(x)
            # print(y_pred)
            y_pred = y_pred[:, -1, :]
            # print(y_pred)
            pred, _ = get_pred_from_outputs(y_pred)
            # print(pred)
            if f'task_{task}' not in self.performance_anytime:
                self.performance_anytime[f'task_{task}'] = {
                    'acc': [],
                    'kappa': [],
                    'predictions': []
                }
            
            self.performance_anytime[f'task_{task}']['predictions'].append(pred)


    def save_final_metrics(self, task):
        """Finalizes and saves metrics for learning."""
     
        # Ensure the dictionary is initialized before using it
        if f'task_{task}' not in self.performance:
            self.performance[f'task_{task}'] = {
                'acc': [],
                'kappa': [],
                'predictions': []
            }

        print(f'Accuracy for Task {task}:', np.mean(self.performance[f'task_{task}']['acc'][0:50]),np.mean(self.performance[f'task_{task}']['acc']))
        print(f'Cohen Kappa for Task {task}:', np.mean(self.performance[f'task_{task}']['kappa'][0:50]),np.mean(self.performance[f'task_{task}']['kappa']))
        print('*****************************************************')
        
    def save_final_metrics_anytime(self,task,y):
        print(task)
        self.performance_anytime[f'task_{task}']['predictions'] = [None] * (self.seq_len-1) + self.performance_anytime[f'task_{task}']['predictions']
        acc = metrics.Accuracy()
        ck = metrics.CohenKappa()
        for i in range(len(y)):
            try:
                self.performance_anytime[f'task_{task}']['acc'].append((copy.deepcopy(acc.update(int(self.performance_anytime[f'task_{task}']['predictions'][i].item()),int(y[i])))).get())
                self.performance_anytime[f'task_{task}']['kappa'].append((copy.deepcopy(ck.update(int(self.performance_anytime[f'task_{task}']['predictions'][i].item()),int(y[i])))).get())
            except:
                self.performance_anytime[f'task_{task}']['acc'].append((copy.deepcopy(acc.update(self.performance_anytime[f'task_{task}']['predictions'][i],int(y[i])))).get())
                self.performance_anytime[f'task_{task}']['kappa'].append((copy.deepcopy(ck.update(self.performance_anytime[f'task_{task}']['predictions'][i],int(y[i])))).get())
            # try:
            #     self.performance_anytime[f'task_{task}']['acc'].append(copy.deepcopy(acc.update(int(self.performance_anytime[f'task_{task}']['predictions'][i]),int(y[i]))))
            #     self.performance_anytime[f'task_{task}']['kappa'].append(copy.deepcopy(ck.update(int(self.performance_anytime[f'task_{task}']['predictions'][i]),int(y[i]))))
            # except:
            #     self.performance_anytime[f'task_{task}']['acc'].append(copy.deepcopy(acc.update(self.performance_anytime[f'task_{task}']['predictions'][i],int(y[i]))))
            #     self.performance_anytime[f'task_{task}']['kappa'].append(copy.deepcopy(ck.update(self.performance_anytime[f'task_{task}']['predictions'][i],int(y[i]))))

        print('All points Accuracy= ', self.performance_anytime[f'task_{task}']['acc'][-1])
        print('All points cohen kappa= ', self.performance_anytime[f'task_{task}']['kappa'][-1])
        
        
    def plot_metrics(self):
        """Plots accuracy and Cohen Kappa metrics."""
        tasks = list(self.performance.keys())
        acc_values = [self.performance[task]['acc'][-1] for task in tasks]
        kappa_values = [self.performance[task]['kappa'][-1] for task in tasks]

        # Accuracy Plot
        plt.figure(figsize=(10, 4))
        plt.plot(tasks, acc_values, marker='o', label="Accuracy", color='b')
        plt.xlabel('Task Number')
        plt.ylabel('Accuracy')
        plt.title('Final Accuracy for Each Task')
        plt.legend()
        plt.grid()
        plt.show()

        # Cohen Kappa Plot
        plt.figure(figsize=(10, 4))
        plt.plot(tasks, kappa_values, marker='s', label="Cohen Kappa", color='r')
        plt.xlabel('Task Number')
        plt.ylabel('Cohen Kappa')
        plt.title('Final Cohen Kappa for Each Task')
        plt.legend()
        plt.grid()
        plt.show()
