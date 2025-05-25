from river import metrics
import numpy as np
import pickle

import torch
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


class cPB_SML:
    def __init__(
        self,
        # this parameter is useless and i should remove it
        hidden_size=50,
        device=None,
        stride: int = 1,
        lr: float = 0.01,
        seq_len: int = 5,
        base_model='GRU',
        pretrain_model_addr='',
        mask_weights=[],
        mask_init='1s',
        number_of_tasks=4,
        epoch_size=10,
        input_size=2,
        many_to_one = False,
        EndOfTask = 188,
        **kwargs,

    ):
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.stride=stride
        self.seq_len=seq_len
        self.lr=lr
        self.hidden_size = hidden_size
        self.base_model=base_model
        self.pretrain_model_addr=pretrain_model_addr
        self.mask_init=mask_init
        self.weights_list=[]
        self.selected_mask_index=[]
        self.epoch_size=epoch_size
        self.input_size=input_size
        self.mask_weights=mask_weights
        self.all_models_weight=[]
        self.performance=dict()
        self.performance[f'task_{1}']={}
        self.performance[f'task_{1}']['acc']=[]
        self.performance[f'task_{1}']['kappa']=[]
        self.acc_saving = [[]]
        self.cohen_kappa_saving=[[]]
        self.mask_selection=True
        self.many_to_one=many_to_one

        if base_model=='GRU':
            self.model = ModifiedRNN(pretrain_model_addr=pretrain_model_addr,hidden_size=self.hidden_size,base_model=base_model,seq_len=seq_len,mask_weights=mask_weights,mask_init=mask_init,input_size=self.input_size,many_to_one=self.many_to_one)
        elif base_model=='LSTM':
            self.model = ModifiedRNN(pretrain_model_addr=pretrain_model_addr,hidden_size=self.hidden_size,base_model=base_model,seq_len=seq_len,mask_weights=mask_weights,mask_init=mask_init,input_size=self.input_size,many_to_one=self.many_to_one)
        self.current_task_index=1
        self.device=device
        self.previous_data_points_anytime_inference = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_batch_train = None
        self.previous_data_points_batch_test = None
        # self.base_learner
        self.masks = []
        self.ensemble = []
        self.ensemble.append(self.model)
        self.selected_model = 0
        self.acc_saving = [[metrics.Accuracy()] for _ in range(len(self.ensemble))]
        self.cohen_kappa_saving = [[metrics.CohenKappa()] for _ in range(len(self.ensemble))]
        self.all_batch_acc=[[] for _ in range(number_of_tasks)]
        self.all_batch_kappa=[[] for _ in range(number_of_tasks)]
        self.performance_CC = [[copy.deepcopy(metrics.CohenKappa())] for _ in range(len(self.ensemble))]
        self.performance_ACC = [[copy.deepcopy(metrics.Accuracy())] for _ in range(len(self.ensemble))]
        self.predictions = [[] for _ in range(0, len(self.ensemble))] # len(ensemble), batch_size
        self.count = 0
        self.EndOfTask = EndOfTask

        self.selected_model_index=[]


    def predict_one(self, x: np.array, previous_data_points: np.array = None):
        x = np.array(x).reshape(1, -1)
        if previous_data_points is not None:
            self.previous_data_points_anytime_inference = previous_data_points
        if self.previous_data_points_anytime_inference is None:
            self.previous_data_points_anytime_inference = x
            for i, model in enumerate(self.ensemble):
              self.predictions[i].append([0])
            return None
        if len(self.previous_data_points_anytime_inference) != self.seq_len - 1:
            self.previous_data_points_anytime_inference = np.concatenate(
                [self.previous_data_points_anytime_inference, x]
            )
            for i, model in enumerate(self.ensemble):
              self.predictions[i].append([0])
            return None
        self.previous_data_points_anytime_inference = np.concatenate(
            [self.previous_data_points_anytime_inference, x]
        )
        x = self._convert_to_tensor_dataset(
            self.previous_data_points_anytime_inference
        )
        self.previous_data_points_anytime_inference = (
            self.previous_data_points_anytime_inference[1:]
        )
            #return int(pred[-1].detach().cpu().numpy())

        for i, model in enumerate(self.ensemble):
            with torch.no_grad():
              self.loss_on_seq=True
              if not self.loss_on_seq:
                  pred, _ = get_pred_from_outputs(self.ensemble[i](x)[0])
              else:
                  pred, _ = get_pred_from_outputs(self.ensemble[i](x))

            self.predictions[i].append(pred)
        #return self.predictions[self.selected_model][-1]

    def learn_many(self, x, y):

        for i, p in enumerate(self.predictions): # iterate on each mask in the ensemble
            for j in range(len(y)): # iterating on each prediction of the mask i

                self.performance_CC[i][0].update(int(p[j][0]), int(y[j]))
                self.performance_ACC[i][0].update(int(p[j][0]), int(y[j]))
                self.acc_saving[i].append(copy.deepcopy(self.performance_ACC[i][0]))
                self.cohen_kappa_saving[i].append(copy.deepcopy(self.performance_CC[i][0]))
                if self.mask_selection==False:
                    self.performance[f'task_{self.current_task_index}']['acc'].append(copy.deepcopy(self.performance_ACC[i][0]))
                    self.performance[f'task_{self.current_task_index}']['kappa'].append(copy.deepcopy(self.performance_CC[i][0]))



        self.selected_model = np.argmax([self.performance_CC[i][0].get() for i in range(len(self.performance_CC))])

        self.count += 1
        if self.count == 50:
            self.mask_selection=False

            self.selected_model_index.append(self.selected_model)
            self.performance[f'task_{self.current_task_index}']['acc']=copy.deepcopy(self.acc_saving[self.selected_model])
            self.performance[f'task_{self.current_task_index}']['kappa']=copy.deepcopy(self.cohen_kappa_saving[self.selected_model])
            self.ensemble = [self.ensemble[self.selected_model]]
            self.performance_CC = [self.performance_CC[self.selected_model]]
            self.performance_ACC = [self.performance_ACC[self.selected_model]]
            self.predictions = [self.predictions[self.selected_model]]
            self.selected_model = 0



        self.predictions=[[] for _ in range(len(self.ensemble))]
        x = np.array(x)
        y = list(y)
        first_batch = False
        self.loss_on_seq=True
        if self.loss_on_seq:
            if self.previous_data_points_batch_train is None:
                first_batch = True
            else:
                x = np.concatenate([x, self.previous_data_points_batch_train], axis=0)
                self.previous_data_points_batch_train = x[-(self.seq_len - 1) :]
        x, y, y_seq = self._load_batch(x, y)
        if first_batch:
            y = y[self.seq_len - 1 :]
        x, y = x.to(self.device), y.to(self.device)

        for i,model in enumerate(self.ensemble):
            for e in range(1, self.epoch_size + 1):

              outputs = model(x)

              if not self.loss_on_seq:
                  print('not self.loss_on_seq:')
                  outputs = get_samples_outputs(outputs)
                  
              print(outputs.shape)
              print(y)
              
              
              loss = self.loss_fn(outputs, y)
              optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
              optimizer.zero_grad()
              loss.backward(retain_graph=True)
              optimizer.step()
              self.ensemble[i]=model
              print(ggggg)

    def get_seq_len(self):
        return self.seq_len

    def predict_many(self, x: np.array, column_id: int = None):
        x = np.array(x)
        if x.shape[0] < self.get_seq_len():
            return np.array([None] * x.shape[0])
        first_train = False
        self.loss_on_seq=True
        if self.loss_on_seq:
            if self.previous_data_points_batch_train is not None:
                x = np.concatenate([x, self.previous_data_points_batch_train], axis=0)
                self.previous_data_points_batch_train = x[-(self.seq_len - 1) :]
            else:
                first_train = True
        x = self._convert_to_tensor_dataset(x).to(self.device)
        with torch.no_grad():
            outputs = self.ensemble[0](x)
            if not self.loss_on_seq:
                outputs = get_samples_outputs(outputs)
            pred, _ = get_pred_from_outputs(outputs)
            pred = pred.detach().cpu().numpy()
            if first_train:
                return np.concatenate(
                    [np.array([None for _ in range(self.seq_len - 1)]), pred], axis=0
                )
            return pred
    def add_new_column(self,task):
        self.mask_selection=True
        self.current_task_index+=1
        self.performance[f'task_{self.current_task_index}']={}
        self.performance[f'task_{self.current_task_index}']['acc']=[]
        self.performance[f'task_{self.current_task_index}']['kappa']=[]
        print('after concept drift')
        print('self.performance_CC',self.performance_CC)
        print('self.performance_ACC',self.performance_ACC)
        param_list=[]
        weights_list=[]
        weights_list=copy.deepcopy(self.ensemble[0].state_dict())
        for params in weights_list:
          param_list.append(params)

        mask_weights=[]
        mask_weights.append(weights_list[param_list[-5]])
        mask_weights.append(weights_list[param_list[-4]])
        mask_weights.append(weights_list[param_list[-3]])
        mask_weights.append(weights_list[param_list[-2]])
        mask_weights.append(weights_list[param_list[-1]])
        self.model=ModifiedRNN(pretrain_model_addr=self.pretrain_model_addr, hidden_size= self.hidden_size,
                              base_model=self.base_model,seq_len=self.seq_len,mask_weights=mask_weights,
                              mask_init=self.mask_init,input_size=self.input_size,many_to_one=self.many_to_one)
        self.masks.append(self.model)
        #self.ensemble=[[] for _ in range(len(self.masks))]
        self.ensemble=[]
        for i in range(len(self.masks)):
          weights_list=[]
          weights_list=copy.deepcopy(self.masks[i].state_dict())
          for params in weights_list:
            param_list.append(params)

          mask_weights=[]
          mask_weights.append(weights_list[param_list[-5]])
          mask_weights.append(weights_list[param_list[-4]])
          mask_weights.append(weights_list[param_list[-3]])
          mask_weights.append(weights_list[param_list[-2]])
          mask_weights.append(weights_list[param_list[-1]])
          self.model=ModifiedRNN(pretrain_model_addr=self.pretrain_model_addr, hidden_size= self.hidden_size,
                                base_model=self.base_model,seq_len=self.seq_len,mask_weights=mask_weights,
                                mask_init=self.mask_init,input_size=self.input_size,many_to_one=self.many_to_one)

          self.ensemble.append(self.model)
        mask_weights=[]
        self.model = ModifiedRNN(pretrain_model_addr=self.pretrain_model_addr,hidden_size=self.hidden_size,base_model=self.base_model,seq_len=self.seq_len,mask_weights=self.mask_weights,mask_init=self.mask_init,input_size=self.input_size,many_to_one=self.many_to_one)

        self.ensemble.append(self.model) # random initialized
        self.acc_saving = [[] for _ in self.ensemble]
        self.cohen_kappa_saving = [[] for _ in self.ensemble]
        self.reset_previous_data_points()
        self.count = 0
        self.performance_CC = [[copy.deepcopy(metrics.CohenKappa())] for _ in range(len(self.ensemble))]
        self.performance_ACC = [[copy.deepcopy(metrics.Accuracy())] for _ in range(len(self.ensemble))]
        self.predictions = [[] for _ in range(len(self.ensemble))] # len(ensemble), batch_size

    def reset_previous_data_points(self):
        self.previous_data_points_batch_train = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_anytime_inference = None

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