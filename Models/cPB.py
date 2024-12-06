import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings
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
from Models.PiggyBack import(
	PBGRU,
  	PBLSTM,
)
from Models.network import ModifiedRNN
import matplotlib.pyplot as plt

class cPB:
    """
    Class that implements the cPNN (Custom Piggyback Neural Network) structure for task-specific learning
    with weight-sharing and mask-based adaptation.

    Parameters
    ----------
    model_class : class, default=PBGRU
        The base model class used in the network (e.g., PBGRU or PBLSTM).
    hidden_size : int, default=50
        Number of hidden units in the model.
    device : torch.device or None, default=None
        Device to run the model ('cpu' or 'cuda').
    stride : int, default=1
        Stride length for splitting sequences.
    lr : float, default=0.01
        Learning rate for training.
    seq_len : int, default=5
        Length of input sequences.
    base_model : str, default='GRU'
        Type of base model to use ('GRU' or 'LSTM').
    pretrain_model_addr : str, default=''
        Path to the pretrained model weights.
    mask_weights : list, default=[]
        Weights used for masks in piggyback models.
    mask_init : str, default='uniform'
        Initialization method for masks.
    number_of_tasks : int, default=4
        Number of tasks to be handled by the model.
    epoch_size : int, default=10
        Number of epochs for training each task.
    input_size : int, default=2
        Number of input features.
    **kwargs : dict
        Additional parameters for model initialization.

    Attributes
    ----------
    model : ModifiedRNN
        The neural network model for handling task-specific weights and masks.
    loss_fn : torch.nn.CrossEntropyLoss
        Loss function used during training.
    weights_list : list
        List of task-specific weights for the model.
    performance : dict
        Dictionary to store metrics (accuracy and Cohen Kappa) for each task.
    """

    def __init__(self, model_class=PBGRU, hidden_size=50, device=None, stride=1, lr=0.01,
                 seq_len=5, base_model='GRU', pretrain_model_addr='', mask_weights=[],
                 mask_init='uniform', number_of_tasks=4, epoch_size=10, input_size=2, **kwargs):
        """
        Initializes the cPB class with parameters for task-specific weight-sharing and mask handling.
        """
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.stride = stride
        self.seq_len = seq_len
        self.lr = lr
        self.hidden_size = hidden_size
        self.base_model = base_model
        self.pretrain_model_addr = pretrain_model_addr
        self.mask_init = mask_init
        self.weights_list = []
        self.selected_mask_index = []
        self.epoch_size = epoch_size
        self.input_size = input_size
        self.all_batch_acc = [[] for _ in range(number_of_tasks)]
        self.all_batch_kappa = [[] for _ in range(number_of_tasks)]
        self.acc_saving = [[]]
        self.cohen_kappa_saving = [[]]
        self.all_models_weight = []
        self.performance = dict()

        # Initialize the model with pretrained weights if available
        if pretrain_model_addr != '':
            self.model = ModifiedRNN(pretrain_model_addr=pretrain_model_addr,
                                     hidden_size=self.hidden_size, base_model=base_model,
                                     seq_len=seq_len, mask_weights=mask_weights,
                                     mask_init=mask_init, input_size=self.input_size)
            self.initial_weights = self.model.state_dict()
        self.all_models_weight.append([])

    def get_seq_len(self):
        """
        Returns the sequence length used in the model.

        Returns
        -------
        int
            The sequence length.
        """
        return self.seq_len

    def _cut_in_sequences(self, x, y):
        """
        Splits the data into sequences of specified length with the given stride.

        Parameters
        ----------
        x : numpy.ndarray
            Input features.
        y : numpy.ndarray or None
            Labels corresponding to the input features.

        Returns
        -------
        seqs_features : numpy.ndarray
            Features split into sequences.
        seqs_targets : numpy.ndarray
            Labels split into sequences.
        """
        seqs_features = []
        seqs_targets = []
        for i in range(0, len(x), self.stride):
            if len(x) - i >= self.seq_len:
                seqs_features.append(x[i: i + self.seq_len, :].astype(np.float32))
                if y is not None:
                    seqs_targets.append(np.asarray(y[i: i + self.seq_len], dtype=np.int_))
        return np.asarray(seqs_features), np.asarray(seqs_targets)

    def _convert_to_tensor_dataset(self, x, y=None):
        """
        Converts the input features and labels into a TensorDataset.

        Parameters
        ----------
        x : numpy.ndarray
            Input features.
        y : numpy.ndarray or None
            Labels corresponding to the input features.

        Returns
        -------
        dataset : torch.utils.data.TensorDataset
            A TensorDataset containing the features and labels.
        """
        x, y = self._cut_in_sequences(x, y)
        x = torch.tensor(x)
        if len(y) > 0:
            y = torch.tensor(y).type(torch.LongTensor)
            return data_utils.TensorDataset(x, y)
        return x

    def learn_many(self, x, y, task_number):
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
        self.model = self.rebuild_model(task_number)
        x = np.array(x)
        y = list(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        x, y, _ = self._load_batch(x, y)
        for _ in range(self.epoch_size):
            optimizer.zero_grad()
            y_pred = self.model(x)
            y_pred = get_samples_outputs(y_pred)
            loss = self.loss_fn(y_pred, y)
            loss.backward(retain_graph=True)
            optimizer.step()
        self.update_weights(task_number)

    def rebuild_model(self, task_number):
        """
        Rebuilds the model for a specific task using the corresponding weights.

        Parameters
        ----------
        task_number : int
            Index of the task to rebuild the model for.

        Returns
        -------
        model : ModifiedRNN
            The rebuilt model with task-specific weights.
        """
        param_list = []
        for params in self.weights_list[task_number]:
            param_list.append(params)

        mask_weights = [
            self.weights_list[task_number][param_list[-5]],
            self.weights_list[task_number][param_list[-4]],
            self.weights_list[task_number][param_list[-3]],
            self.weights_list[task_number][param_list[-2]],
            self.weights_list[task_number][param_list[-1]],
        ]
        self.model = ModifiedRNN(pretrain_model_addr=self.pretrain_model_addr,
                                 hidden_size=self.hidden_size, base_model=self.base_model,
                                 seq_len=self.seq_len, mask_weights=mask_weights,
                                 mask_init=self.mask_init, input_size=self.input_size)
        return self.model

    def predict_many(self, x, y, mask_number, task_number, mask_selection=False):
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
            self.model = self.rebuild_model(mask_number)
            x = np.array(x)
            y = list(y)
            x, y, _ = self._load_batch(x, y)
            y_pred = self.model(x)
            y_pred = get_samples_outputs(y_pred)
            pred, _ = get_pred_from_outputs(y_pred)
            kappa = cohen_kappa(y, pred).item()
            acc = accuracy_score(np.array(y), np.array(pred))
            self.acc_saving[mask_number].append(acc)
            self.cohen_kappa_saving[mask_number].append(kappa)
            if not mask_selection:
                self.performance[f'task_{task_number}']['acc'].append(acc)
                self.performance[f'task_{task_number}']['kappa'].append(kappa)

    def plotting(self):
        """
        Plots the cumulative accuracy and Cohen Kappa for all tasks.

        Displays
        -------
        - A line plot showing cumulative accuracy over batches.
        - A line plot showing cumulative Cohen Kappa over batches.
        """
        x0 = np.cumsum(self.all_batch_acc[0]) / np.arange(1, len(self.all_batch_acc[0]) + 1)
        x1 = np.cumsum(self.all_batch_acc[1]) / np.arange(1, len(self.all_batch_acc[1]) + 1)
        x2 = np.cumsum(self.all_batch_acc[2]) / np.arange(1, len(self.all_batch_acc[2]) + 1)
        x3 = np.cumsum(self.all_batch_acc[3]) / np.arange(1, len(self.all_batch_acc[3]) + 1)
        all_x = np.concatenate((x0, x1, x2, x3), axis=0)
        vertical_lines_x = [len(x0), len(x0) + len(x1), len(x0) + len(x1) + len(x2)]
        y = all_x
        x = list(range(1, len(all_x) + 1))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.margins(x=0.0)
        for i in vertical_lines_x:
            plt.axvline(x=i, color='#D3D3D3', linestyle='-')
        ax.plot(x, y, color='#ff1d58')
        plt.xlabel('Batch Number')
        plt.ylabel('Cumulative Accuracy')
        plt.title('Cumulative Accuracy Over Batches')
        plt.legend()
        plt.show()

        x0 = np.cumsum(self.all_batch_kappa[0]) / np.arange(1, len(self.all_batch_kappa[0]) + 1)
        x1 = np.cumsum(self.all_batch_kappa[1]) / np.arange(1, len(self.all_batch_kappa[1]) + 1)
        x2 = np.cumsum(self.all_batch_kappa[2]) / np.arange(1, len(self.all_batch_kappa[2]) + 1)
        x3 = np.cumsum(self.all_batch_kappa[3]) / np.arange(1, len(self.all_batch_kappa[3]) + 1)
        all_x = np.concatenate((x0, x1, x2, x3), axis=0)
        y = all_x
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.margins(x=0.0)
        for i in vertical_lines_x:
            plt.axvline(x=i, color='#D3D3D3', linestyle='-')
        ax.plot(x, y, color='#ff1d58')
        plt.xlabel('Batch Number')
        plt.ylabel('Cumulative Cohen Kappa')
        plt.title('Cumulative Cohen Kappa Over Batches')
        plt.legend()
        plt.show()
