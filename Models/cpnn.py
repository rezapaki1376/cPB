import pickle

import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from river import metrics
import warnings

from Models.cpnn_columns import cPNNColumns
from utils.utils import (
    customized_loss,
    accuracy,
    cohen_kappa,
    kappa_temporal,
    get_samples_outputs,
    get_pred_from_outputs,
    kappa_temporal_score,
)
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from Models.clstm import (
    cLSTMLinear,
)


class cPNN:
    """
    Class that implements all the cPNN structure.
    """

    def __init__(
        self,
        column_class=cLSTMLinear,
        device=None,
        lr: float = 0.01,
        seq_len: int = 5,
        stride: int = 1,
        train_epochs: int = 10,
        train_verbose: bool = False,
        anytime_learner: bool = False,
        acpnn: bool = False,
        qcpnn: bool = False,
        initial_task_id: int = 1,
        batch_size: int = 128,
        save_column_freq: int = None,
        save_last_n_columns: int = 5,
        drift_delay: int = 5 * 10**3,
        **kwargs,
    ):
        """
        Parameters
        ----------
        column_class: default: cLSTMLinear.
            The class that implements the column.
        device: default: None.
            Torch's device, if None its value is set to 'cpu'.
        lr: float, default: 0.01.
            The learning rate value of single columns' Adam Optimizer.
        seq_len: int, default: 5.
            The length of the sliding window that builds the single sequences.
        stride: int, default: 1.
            The length of the sliding window's stride.
        train_epochs: int, default: 10.
            In case of anytime_learner=False, the training epochs to perform in learn_many method.
        train_verbose: bool, default:False.
            True if, during the learn_many execution, you want to print the metrics after each training epoch.
        anytime_learner: bool, default: False.
            If True the model learns data point by data point by data point.
            Otherwise, it learns batch by batch.
        acpnn: bool, default: False.
            In case of anytime_learner = False is always True.
            If True, the model is A-cPNN. It considers only past temporal dependencies. The model
            is a many_to_one model and each data point's prediction is associated to the first sequence in which
            it appears.
            If False, the model considers both past and future temporal dependencies.The model is a many_to_many model
            and each data point's prediction is the average prediction between all the sequences in which it appears.
        qcpnn: bool, default: False
            If True the model is a Q-cPNN. After a concept drift, the column is quantized.
        initial_task_id: int, default: 1.
            The id of the first task.
        batch_size: int, default: 128.
            The training batch size. If you call learn_one method, the model will accumulate batch_size data point
            before performing the training.
        save_column_freq, int, default: None
            The frequency at which the model stores the current state of the last column. Since the drift detector
            may have a delay, before adding a new column, the stored state of the last column is restored. This avoids
            to use a column that does not represent the task.
        kwargs:
            Parameters of column_class.
        """
        self.anytime_learner = anytime_learner
        if self.anytime_learner:
            self.loss_on_seq = True
            self.many_to_one = True
        else:
            self.loss_on_seq = acpnn
            if acpnn:
                self.many_to_one = True
            else:
                self.many_to_one = False

        self.columns_args = kwargs
        self.columns_args["column_class"] = column_class
        self.columns_args["device"] = device
        self.columns_args["lr"] = lr
        self.columns_args["many_to_one"] = self.many_to_one
        self.columns_args["quantize"] = qcpnn
        self.columns_args["batch_size"] = batch_size
        self.columns = cPNNColumns(**self.columns_args)
        self.seq_len = seq_len
        self.stride = stride
        self.train_epochs = train_epochs
        self.train_verbose = train_verbose
        self.columns_perf = [metrics.CohenKappa()]
        self.task_ids = [initial_task_id]
        self.previous_data_points_anytime_inference = None
        self.previous_data_points_anytime_hidden = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_batch_train = None
        self.previous_data_points_batch_inference = None
        self.x_batch = []
        self.y_batch = []
        self.batch_size = batch_size
        self.save_column_freq = save_column_freq
        self.saved_columns = []
        self.save_last_n_columns = save_last_n_columns
        self.last_prediction = -1
        self.drift_delay = drift_delay
        self.cont = 1
        self.train_cont = [0]

    def get_seq_len(self):
        return self.seq_len

    def set_initial_task(self, task):
        self.task_ids = [task]

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

    def add_new_column(self, task_id=None):
        """
        It adds a new column to the cPNN architecture, after a concept drift.

        Parameters
        ----------
        task_id: int, default: None
            The id of the new task. If None it increments the last one.
        """
        if len(self.saved_columns) > 0:
            cols = [
                sc
                for sc in self.saved_columns
                if sc["cont"] <= self.cont - self.drift_delay
            ]
            if len(cols) == 0:
                col = self.saved_columns[0]
            else:
                col = cols[-1]
            self.columns.columns[-1] = pickle.loads(pickle.dumps(col["column"]))
            self.saved_columns = []
            print(f"Restored column:{col['cont']}, Current cont: {self.cont}")
        self.reset_previous_data_points()
        self.last_prediction = -1
        self.columns.add_new_column()
        self.columns_perf.append(metrics.CohenKappa())
        self.train_cont.append(0)
        self.x_batch = []
        self.y_batch = []
        if task_id is None:
            self.task_ids.append(self.task_ids[-1] + 1)
        else:
            self.task_ids.append(task_id)

    def learn_one(self, x: np.array, y: int, previous_data_points: np.array = None):
        """
        It trains cPNN on a single data point. In the case of anytime learner, the training is performed after each
        data point. In the case of periodic learner (anytime_learner=False), the training is performed after filling
        up a mini_batch containing batch_size data points.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the single data point.
        y: int
            The target value of the single data point.
        previous_data_points: numpy.array, default: None.
            The features value of the data points preceding x in the sequence.
            If None, it uses the last seq_len-1 points seen during the last calls of the method.
            It returns None if the model has not seen yet seq_len-1 data points and previous_data_points is None.
        """
        if self.last_prediction != -1:
            self.columns_perf[-1].update(y, self.last_prediction)
        if not self.anytime_learner:
            self.x_batch.append(x)
            self.y_batch.append(y)
            self.cont += 1
            if len(self.x_batch) == self.batch_size:
                self.learn_many(np.array(self.x_batch), np.array(self.y_batch))
                self.x_batch = []
                self.y_batch = []
            if self.save_column_freq is not None:
                if self.cont % self.save_column_freq == 0:
                    self.saved_columns.append(
                        {
                            "cont": self.cont,
                            "column": pickle.loads(
                                pickle.dumps(self.columns.columns[-1])
                            ),
                        }
                    )
                    self.saved_columns = self.saved_columns[-4:]
            return None
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1, -1)
        if previous_data_points is not None:
            self.previous_data_points_anytime_train = previous_data_points
        if self.previous_data_points_anytime_train is None:
            self.previous_data_points_anytime_train = x
            return None
        if len(self.previous_data_points_anytime_train) != self.seq_len - 1:
            self.previous_data_points_anytime_train = np.concatenate(
                [self.previous_data_points_anytime_train, x]
            )
            return None
        self.previous_data_points_anytime_train = np.concatenate(
            [self.previous_data_points_anytime_train, x]
        )
        x, y, _ = self._load_batch(self.previous_data_points_anytime_train, y)
        self._fit(x, y.view(-1))
        self.previous_data_points_anytime_train = (
            self.previous_data_points_anytime_train[1:]
        )

    def learn_many(self, x: np.array, y: np.array) -> dict:
        """
        It trains cPNN on a single mini-batch of data points.
        In the case of not acpnn, it computes the loss after averaging each sample's predictions.
        *ONLY FOR PERIODIC LEARNER (anytime_learner=False)*

        Parameters
        ----------
        x: numpy.array or list
            The features values of the mini-batch.
        y: np.array or list
            The target values of the mini-batch.

        Returns
        -------
        perf_train: dict
            The dictionary representing training's performance. Each key contains the list representing all the epochs' performances.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
            For each metric the dict contains a list of epochs' values.
        """
        if self.anytime_learner:
            warnings.warn(
                "The model is an anytime learner, it cannot learn from batch.\n"
                + "Loop on learn_one method to learn from multiple data points"
            )
            return {}
        x = np.array(x)
        y = list(y)
        if x.shape[0] < self.get_seq_len():
            return {}

        if self.loss_on_seq:
            if self.previous_data_points_batch_train is not None:
                x = np.concatenate([self.previous_data_points_batch_train, x], axis=0)
                y = np.concatenate([[i for i in range(self.seq_len - 1)], y], axis=0)
            self.previous_data_points_batch_train = x[-(self.seq_len - 1) :]
        x, y, _ = self._load_batch(x, y)
        if self.loss_on_seq:
            y = y[self.seq_len - 1 :]

        perf_train = {
            "accuracy": [],
            "loss": [],
            "kappa": [],
            "kappa_temporal": [],
        }
        for e in range(1, self.train_epochs + 1):
            perf_epoch = self._fit(x, y)
            if self.train_verbose:
                print(
                    "Training epoch ",
                    e,
                    "/",
                    self.train_epochs,
                    ". accuracy: ",
                    perf_epoch["accuracies"],
                    ", loss:",
                    perf_epoch["losses"],
                    sep="",
                    end="\r",
                )
            for k in perf_epoch:
                perf_train[k].append(perf_epoch[k])
        if self.train_verbose:
            print()
            print()

        self.train_cont[-1] = self.train_cont[-1]+1
        return perf_train

    def predict_many(self, x: np.array, column_id: int = None):
        """
        It performs prediction on a single batch.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the batch.
        column_id: int, default: None.
            The id of the column to use. If None the last column is used.

        Returns
        -------
        predictions: numpy.array
            The 1D numpy array (with length batch_size) containing predictions of all samples.
        """
        if self.anytime_learner:
            if self.anytime_learner:
                warnings.warn(
                    "The model is an anytime learner, it cannot predict a batch of data.\n"
                    + "Loop on predict_one method to predict on multiple data points"
                )
                return None
        x = np.array(x)
        if x.shape[0] < self.get_seq_len():
            return np.array([None] * x.shape[0])
        first_train = False
        if self.loss_on_seq:
            if self.previous_data_points_batch_inference is not None:
                x = np.concatenate(
                    [self.previous_data_points_batch_inference, x], axis=0
                )
            else:
                first_train = True
            self.previous_data_points_batch_inference = x[-(self.seq_len - 1) :]
        x = self._convert_to_tensor_dataset(x).to(self.columns.device)
        with torch.no_grad():
            outputs = self.columns(x, column_id)
            if not self.loss_on_seq:
                outputs = get_samples_outputs(outputs)
            pred, _ = get_pred_from_outputs(outputs)
            pred = pred.detach().cpu().numpy()
            if first_train:
                return np.concatenate(
                    [np.array([None for _ in range(self.seq_len - 1)]), pred], axis=0
                )
            return pred

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

    def predict_one(
        self, x: np.array, column_id: int = None, previous_data_points: np.array = None
    ):
        """
        It performs prediction on a single data point.

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
        x = self._single_data_point_prep(x, previous_data_points)
        if x is None:
            self.last_prediction = None
            return None
        with torch.no_grad():
            if not self.loss_on_seq:
                pred, _ = get_pred_from_outputs(self.columns(x, column_id)[0])
            else:
                pred, _ = get_pred_from_outputs(self.columns(x, column_id))
            self.last_prediction = int(pred[-1].detach().cpu().numpy())
        return self.last_prediction

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

    def get_n_columns(self):
        return len(self.columns.columns)

    def reset_previous_data_points(self):
        self.previous_data_points_batch_train = None
        self.previous_data_points_batch_inference = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_anytime_inference = None
        self.previous_data_points_anytime_hidden = None

    def pretraining(
        self, x: np.array, y: list, epochs: int = 100, batch_size: int = 128
    ) -> dict:
        """
        It performs the pretraining on a pretraining set.
        *ONLY FOR BATCH LEARNER*

        Parameters
        ----------
        x: numpy.array
            The features values of the set.
        y: list
            The target values of the set.
        epochs: int, default: 100.
            The number of training epochs to perform on the set.
        batch_size: int, default: 128.
            The training batch size.

        Returns
        -------
        perf_train: dict
            The dictionary representing training's performance.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
            For each metric the dict contains a list of shape (epochs, n_batches) where n_batches is the training
            batches number.
        """
        if self.anytime_learner:
            warnings.warn(
                "The model is an anytime learner, it cannot learn from batch.\n"
                + "You cannot call this method"
            )
            return {}

        perf_train = {
            "accuracy": [],
            "loss": [],
            "kappa": [],
            "kappa_temporal": [],
        }

        x = torch.tensor(x)
        y = torch.tensor(y).type(torch.LongTensor)
        data = data_utils.TensorDataset(x, y)
        loader = DataLoader(data, batch_size=batch_size, drop_last=False)
        print("Pretraining")
        for e in range(1, epochs + 1):
            for k in perf_train:
                perf_train[k].append([])
            for id_batch, (x, y) in enumerate(loader):
                print(
                    f"{id_batch+1}/{len(loader)} batch of {e}/{epochs} epoch", end="\r"
                )
                x, y_seq = self._cut_in_sequences_tensors(x, y)
                perf_batch = self._fit(x, y)
                for k in perf_batch:
                    perf_train[k][-1].append(perf_batch[k])
        print()
        print()
        return perf_train

    def remove_last_column(self):
        if len(self.columns.columns) > 1:
            self.columns_perf = self.columns_perf[:-1]
            self.columns_perf[-1] = metrics.CohenKappa()
            self.task_ids = self.task_ids[:-1]
            self.reset_previous_data_points()
            self.columns.remove_last_column()
            self.unfreeze_last_column()
            self.train_cont = self.train_cont[:-2] + [0]


    def set_quantized(self, quantized):
        self.columns.quantize = quantized
        for column in self.columns.columns:
            column.quantized = quantized

    def unfreeze_last_column(self):
        self.columns.unfreeze_last_column()

    def take_first_columns(self, num_columns: int):
        self.columns_perf = self.columns_perf[:num_columns]
        self.columns_perf[-1] = metrics.CohenKappa()
        self.task_ids = self.task_ids[:num_columns]
        self.reset_previous_data_points()
        self.columns.take_first_columns(num_columns)
        self.train_cont = self.train_cont[:num_columns-1] + [0]

    def _fit(self, x, y):
        x, y = x.to(self.columns.device), y.to(self.columns.device)
        outputs = self.columns(x, train=True)
        if not self.loss_on_seq:
            outputs = get_samples_outputs(outputs)
        loss = customized_loss(outputs, y, self.columns.criterion)
        self.columns.optimizers[-1].zero_grad()
        loss.backward()
        self.columns.optimizers[-1].step()
        outputs = self.columns(x)
        if not self.loss_on_seq:
            outputs = get_samples_outputs(outputs)
        perf_train = {
            "loss": loss.item(),
            "accuracy": accuracy(outputs, y).item(),
            "kappa": cohen_kappa(outputs, y, device=self.columns.device).item(),
        }
        return perf_train

    def get_hidden(self, x, column_id=None):
        return self.columns.get_hidden(x, column_id)
