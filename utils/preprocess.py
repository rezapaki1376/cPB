import pickle

import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from river import metrics
import warnings

from models.utils import (
    customized_loss,
    accuracy,
    cohen_kappa,
    get_samples_outputs,
    get_pred_from_outputs,
)
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from models.clstm import (
    cLSTMLinear,
)


class Preprocess:
    """
    Class that implements all the cPNN structure.
    """
    def __init__(
        self,
        seq_len: int = 10,
    ):
        """
        Parameters
        seq_len: int, default: 5.
            The length of the sliding window that builds the single sequences.
        """
        self.seq_len = seq_len

    def get_seq_len(self):
        return self.seq_len

    def _cut_in_sequences(self, x, y):
        seqs_features = []
        seqs_targets = []
        for i in range(0, len(x), self.stride):
            if len(x) - i >= self.seq_len:
                seqs_features.append(x[i : i + self.seq_len, :].astype(np.float32))
                if y is not None:
                    seqs_targets.append(
                        np.asarray(y[i : i + self.seq_len], dtype=np.int_)
                    )
        return np.asarray(seqs_features), np.asarray(seqs_targets)

    def _cut_in_sequences_tensors(self, x, y):
        seqs_features = []
        seqs_targets = []
        for i in range(0, x.size()[0], self.stride):
            if x.size()[0] - i >= self.seq_len:
                seqs_features.append(
                    x[i : i + self.seq_len, :].view(1, self.seq_len, x.size()[1])
                )
                seqs_targets.append(y[i : i + self.seq_len].view(1, self.seq_len))
        seq_features = torch.cat(seqs_features, dim=0)
        seqs_targets = torch.cat(seqs_targets, dim=0)
        return seq_features, seqs_targets

    def _convert_to_tensor_dataset(self, x, y=None):
        """
        It converts the dataset in order to be inputted to cPNN, by building the different sequences and
        converting them to TensorDataset.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: list, default: None
            The target values of the batch. If None only features will be loaded.
        Returns
        -------
        dataset: torch.data_utils.TensorDataset
            The tensor dataset representing the different sequences.
            The features values have shape: (batch_size - seq_len + 1, seq_len, n_features)
            The target values have shape: (batch_size - seq_len + 1, seq_len)
        """
        x, y = self._cut_in_sequences(x, y)
        x = torch.tensor(x)
        if len(y) > 0:
            y = torch.tensor(y).type(torch.LongTensor)
            return data_utils.TensorDataset(x, y)
        return x

    def _load_batch(self, x: np.array, y: np.array = None):
        """
        It transforms the batch in order to be inputted to cPNN, by building the different sequences and
        converting them to tensors.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: list, default: None.
            The target values of the batch. If None only features will be loaded.
        Returns
        -------
        x: torch.Tensor
            The features values of the created sequences. It has shape: (batch_size - seq_len + 1, seq_len, n_features)
        y: torch.Tensor
            The target values of the samples in the batc. It has length: batch_size. If y is None it returns None.
        y_seq: torch.Tensor
            The target values of the created sequences. It has shape: (batch_size - seq_len + 1, seq_len). If y is None it returns None.
        """
        batch = self._convert_to_tensor_dataset(x, y)
        batch_loader = DataLoader(
            batch, batch_size=batch.tensors[0].size()[0], drop_last=False
        )
        y_seq = None
        for x, y_seq in batch_loader:  # only to take x and y from loader
            break
        y = torch.tensor(y)
        return x, y, y_seq


    def _single_data_point_prep(
        self, x, previous_data_points_param: np.array = None, inference=True
    ):
        x = np.array(x).reshape(1, -1)
        if inference:
            previous_data_points = self.previous_data_points_anytime_inference
        else:
            previous_data_points = self.previous_data_points_anytime_hidden
        if previous_data_points_param is not None:
            previous_data_points = previous_data_points_param
        if previous_data_points is None:
            previous_data_points = x
            if inference:
                self.previous_data_points_anytime_inference = previous_data_points
            else:
                self.previous_data_points_anytime_hidden = previous_data_points
            return None
        if len(previous_data_points) != self.seq_len - 1:
            previous_data_points = np.concatenate([previous_data_points, x])
            if inference:
                self.previous_data_points_anytime_inference = previous_data_points
            else:
                self.previous_data_points_anytime_hidden = previous_data_points
            return None
        previous_data_points = np.concatenate([previous_data_points, x])
        x = self._convert_to_tensor_dataset(previous_data_points).to(
            self.columns.device
        )
        previous_data_points = previous_data_points[1:]
        if inference:
            self.previous_data_points_anytime_inference = previous_data_points
        else:
            self.previous_data_points_anytime_hidden = previous_data_points
        return x


    def get_latent_representation(
        self, x: np.array, column_id: int = None, previous_data_points: np.array = None
    ):
        """
        It gets the hidden layer's output of a specific column given a single data point.
        It returns only the output associated with the last item of the sequence.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the single data point.
        column_id: int, default: None.
            The id of the column to use. If None the last column is used.
        previous_data_points: numpy.array, default: None.
            The features value of the data points preceding x in the sequence.
            If None, it uses the last seq_len-1 points seen during the last calls of the method.
            It returns None if the model has not seen yet seq_len-1 data points and previous_data_points is None.
        Returns
        -------
        prediction : int
            The predicted int label of x.
        """
        x = self._single_data_point_prep(x, previous_data_points, inference=False)
        if x is None:
            return None
        with torch.no_grad():
            h = self.columns.forward_hidden(x, column_id)
            if h is None:
                return h
            return h[0, -1, :].detach().cpu().numpy()