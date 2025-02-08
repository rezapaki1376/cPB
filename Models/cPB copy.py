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
    Class implementing a continual learning framework using Custom Piggyback Neural Networks (cPNN) 
    with weight-sharing, mask-based adaptation, and handling of concept drift.

    Parameters
    ----------
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
    model_type : str, default='CPB'
        Model type, either 'CPB' (Continuous Piggyback) or 'CSS' (Continuous SupSup).
    mask_option : str, default='SUM'
        How masks are applied, either 'SUM' (sum of masks with weights) or 'DOT' (dot product).
    low_rank : bool, default=False
        Whether to apply low-rank factorization to mask matrices.
    **kwargs : dict
        Additional parameters for model initialization.

    Attributes
    ----------
    model : ModifiedRNN
        The neural network model with piggyback masking.
    loss_fn : torch.nn.CrossEntropyLoss
        Loss function used for training.
    weights_list : list
        Stores masks corresponding to different data drifts.
    selected_mask_index : list
        Indexes of the best-performing masks during training.
    performance : dict
        Stores metrics (accuracy, Cohen Kappa) per task.
    all_models_weight : list
        Stores model weights over different drifts.
    concept_drift_masks : list
        Stores masks saved at each concept drift point.
    """

    def __init__(self, hidden_size=50, device=None, stride=1, lr=0.01,
                 seq_len=5, base_model='GRU', pretrain_model_addr='', mask_weights=[],
                 mask_init='1s', number_of_tasks=4, epoch_size=10, batch_first=True, input_size=2,
                 model_type='CPB', mask_option='DOT', low_rank=False, **kwargs):
        """
        Initializes the continual learning framework with piggyback masking and concept drift handling.
        """
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.stride = stride
        self.seq_len = seq_len
        self.lr = lr
        self.hidden_size = hidden_size
        self.device = device
        self.base_model = base_model
        self.pretrain_model_addr = pretrain_model_addr
        self.mask_init = mask_init
        self.model_type = model_type
        self.mask_option = mask_option
        self.low_rank = low_rank
        self.weights_list = []  # Stores masks from different concept drifts
        self.selected_mask_index = []  # Stores best-performing mask index after 50 batches
        self.epoch_size = epoch_size
        self.input_size = input_size
        self.batch_first = batch_first
        self.all_batch_acc = [[] for _ in range(number_of_tasks)]
        self.all_batch_kappa = [[] for _ in range(number_of_tasks)]
        self.acc_saving = [[]]
        self.cohen_kappa_saving = [[]]
        self.all_models_weight = []  # Stores weights over time
        self.performance = dict()
        self.concept_drift_masks = []  # Stores masks at each drift point

        # Initialize the model using pretrained weights if available
        if pretrain_model_addr != '':
            self.model = ModifiedRNN(pretrain_model_addr=pretrain_model_addr,
                                     hidden_size=self.hidden_size, base_model=base_model,
                                     seq_len=seq_len, mask_weights=mask_weights,
                                     mask_init=mask_init, input_size=self.input_size,
                                     batch_first = self.batch_first, model_type = self.model_type, 
                                     mask_option = self.mask_option, low_rank = self.low_rank)
            
        self.initial_weights = self.model.state_dict()
        
        self.all_models_weight.append([])  # Store the initial model weights
        self.final_weights=[]

        # At first concept drift, save the mask and reinitialize
        self.current_mask = None


    def get_seq_len(self):
        """
        Returns the sequence length used in the model.

        Returns
        -------
        int
            The sequence length.
        """
        return self.seq_len

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
    
    def _convert_to_tensor_dataset(self, x, y=None):

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


    def update_weights(self,task_number):
      self.weights_list[task_number]=copy.deepcopy(self.model.state_dict())
      
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
        task_weights = copy.deepcopy(self.weights_list[task_number])
        # mask_weights = [
        # task_weights[-5],  # Extracting from the last 5 elements
        # task_weights[-4],
        # task_weights[-3],
        # task_weights[-2],
        # task_weights[-1],
        # ]
        param_list = []
        for params in self.weights_list[task_number]:
            param_list.append(params)

        # print('param_list',param_list)
        # print(task_weights)
        mask_weights = [
            task_weights[param_list[-5]],
            task_weights[param_list[-4]],
            task_weights[param_list[-3]],
            task_weights[param_list[-2]],
            task_weights[param_list[-1]],
        ]
        self.model = ModifiedRNN(pretrain_model_addr=self.pretrain_model_addr,
                                 hidden_size=self.hidden_size, base_model=self.base_model,
                                 seq_len=self.seq_len, mask_weights=mask_weights,
                                 mask_init=self.mask_init, input_size=self.input_size,
                                 batch_first= self.batch_first, model_type = self.model_type, 
                                     mask_option = self.mask_option, low_rank = self.low_rank)
    
        return self.model

    
    def _load_batch(self, x: np.array, y: np.array = None):

        batch = self._convert_to_tensor_dataset(x, y)
        batch_loader = DataLoader(
            batch, batch_size=batch.tensors[0].size()[0], drop_last=False
        )
        y_seq = None
        for x, y_seq in batch_loader:  # only to take x and y from loader
            break
        y = torch.tensor(y)
        return x, y, y_seq
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

    def add_new_column(self,task_number):
      avg_acc= np.mean(self.acc_saving, axis=1)
      avg_cohen_kappa = np.mean(self.cohen_kappa_saving, axis=1)
      index_of_best_acc = np.argmax(avg_cohen_kappa)
      self.selected_mask_index.append(index_of_best_acc)

      self.performance[f'task_{task_number}']['acc']=self.acc_saving[index_of_best_acc]
      self.performance[f'task_{task_number}']['kappa']=self.cohen_kappa_saving[index_of_best_acc]

      print('list of accuracies that used for evaluating and selecting the models = ',avg_acc)
      print('list of kappa values that used for evaluating and selecting the models = ',avg_cohen_kappa)
      print('index of selcted mask for this task',index_of_best_acc)
      return index_of_best_acc
  
    def save_final_metrics(self,task,best_mask_index):
      self.all_batch_acc[task-1] = copy.deepcopy(self.acc_saving[best_mask_index])
      self.all_batch_kappa[task-1] = copy.deepcopy(self.cohen_kappa_saving[best_mask_index])
      print('All batches Accuracy= ', np.mean(self.all_batch_acc[task-1]))
      print('All batches cohen kappa= ', np.mean(self.all_batch_kappa[task-1]))
      self.acc_saving = [[] for _ in range(task+1)]
      self.cohen_kappa_saving = [[] for _ in range(task+1)]
      
      
    def final_weights_saving(self):
      self.final_weights.append(copy.deepcopy(self.model.state_dict()))
      self.all_models_weight.append([])
      mask_weights=copy.deepcopy(self.model.state_dict())
      param_list=[]
      for params in mask_weights:
        param_list.append(params)
      mask_weights=[
          (param_list[-5],mask_weights[param_list[-5]]),
          (param_list[-4],mask_weights[param_list[-4]]),
          (param_list[-3],mask_weights[param_list[-3]]),
          (param_list[-2],mask_weights[param_list[-2]]),
          (param_list[-1],mask_weights[param_list[-1]]),
      ]
      self.all_models_weight[-1]=[mask_weights]
      
    def weights_copy(self, task_number):
      self.weights_list = []
      for i in range(0,task_number-1):
        self.weights_list.append(copy.deepcopy(self.final_weights[i]))
      self.weights_list.append(copy.deepcopy(self.initial_weights))

      self.performance[f'task_{task_number}']={}
      self.performance[f'task_{task_number}']['acc']=[]
      self.performance[f'task_{task_number}']['kappa']=[]
      
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
