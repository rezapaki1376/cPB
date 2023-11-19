import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings

from models.cpnn_columns import cPNNColumns
from models.utils import (
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
from models.PiggyBack import(
	PiggyBackGRU,
  	PiggyBackLSTM,
)

from models.network import ModifiedRNN

import matplotlib.pyplot as plt
class cPB:
    """
    Class that implements all the cPNN structure.
    """

    def __init__(
        self,
        # this parameter is useless and i should remove it
        model_class=PiggyBackGRU,
        hidden_size=50,
        device=None,
        stride: int = 1,
        lr: float = 0.01,
        seq_len: int = 5,
        base_model='gru',
        pretrain_model_addr='',
        mask_weights=[],
        mask_init='1s',
        number_of_tasks=4,
        epoch_size=5,
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
      self.all_batch_acc=[[] for _ in range(number_of_tasks)]
      self.all_batch_kappa=[[] for _ in range(number_of_tasks)]
      self.acc_saving = [[]]
      self.cohen_kappa_saving=[[]]
      print('hidden_size',hidden_size)

      if model_class==PiggyBackGRU and pretrain_model_addr!='':
        self.model = ModifiedRNN(pretrain_model_addr=pretrain_model_addr,hidden_size=self.hidden_size,base_model=base_model,seq_len=seq_len,mask_weights=mask_weights,mask_init=mask_init)
        self.initial_weights = self.model.state_dict()

      self.final_weights=[]

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

        x, y = self._cut_in_sequences(x, y)
        x = torch.tensor(x)
        if len(y) > 0:
            y = torch.tensor(y).type(torch.LongTensor)
            return data_utils.TensorDataset(x, y)
        return x

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


    def learn_many(self,x,y,task_number):

      self.model=self.rebuild_model(task_number)

      x = np.array(x)
      y = list(y)

      optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
      x, y, _ = self._load_batch(x, y)
      #print('inside the train and fit', x.shape)
      for i in range(0,self.epoch_size):
        y_pred = self.model(x)
        y_pred = get_samples_outputs(y_pred)
        pred, _ = get_pred_from_outputs(y_pred)
        loss = self.loss_fn(y_pred, y)
        # backward pass
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # update weights
        optimizer.step()
      self.update_weights(task_number)

    def mask_return(self):
      return self.model.state_dict()
    def update_weights(self,task_number):
      self.weights_list[task_number]=copy.deepcopy(self.model.state_dict())

    def rebuild_model(self,task_number):
      param_list=[]
      for params in self.weights_list[task_number]:
        param_list.append(params)

      mask_weights=[]
      mask_weights.append(self.weights_list[task_number][param_list[-5]])
      mask_weights.append(self.weights_list[task_number][param_list[-4]])
      mask_weights.append(self.weights_list[task_number][param_list[-3]])
      mask_weights.append(self.weights_list[task_number][param_list[-2]])
      mask_weights.append(self.weights_list[task_number][param_list[-1]])
      self.model=ModifiedRNN(pretrain_model_addr=self.pretrain_model_addr, hidden_size= self.hidden_size,
                             base_model=self.base_model,seq_len=self.seq_len,mask_weights=mask_weights,
                             mask_init=self.mask_init)
      return self.model

    def final_weights_saving(self):
      self.final_weights.append(copy.deepcopy(self.model.state_dict()))

    def predict_many(self,x,y,task_number):
      x = np.array(x)
      y = list(y)
      x, y, _ = self._load_batch(x, y)
      #print('input shape', x.shape)
      y_pred = self.model(x)
      y_pred = get_samples_outputs(y_pred)
      pred, _ = get_pred_from_outputs(y_pred)
      kappa=cohen_kappa(y,pred).item()
      acc=accuracy_score(np.array(y),np.array(pred))
      self.acc_saving[task_number].append(acc)
      self.cohen_kappa_saving[task_number].append(kappa)
      #return acc, kappa

    def initial_weights_returning(self):
      return self.initial_weights

    def weights_copy(self, task_number):
      weights_list=[]
      for i in range(0,task_number-1):
        self.weights_list.append(copy.deepcopy(self.final_weights[i]))
      self.weights_list.append(copy.deepcopy(self.initial_weights))

    def add_new_column(self):
      avg_acc= np.mean(self.acc_saving, axis=1)
      avg_cohen_kappa = np.mean(self.cohen_kappa_saving, axis=1)
      index_of_best_acc = np.argmax(avg_acc)
      self.selected_mask_index.append(index_of_best_acc)
      print('list of accuracies that used for evaluating and selecting the models = ',avg_acc)
      print('list of kappa values that used for evaluating and selecting the models = ',avg_cohen_kappa)
      print('index of selcted mask for this task',index_of_best_acc)
      return index_of_best_acc
    def save_final_metrics(self,task,best_mask_index):
      self.all_batch_acc[task-1]=copy.deepcopy(self.acc_saving[best_mask_index])
      self.all_batch_kappa[task-1]=copy.deepcopy(self.cohen_kappa_saving[best_mask_index])
      print('All batches Accuracy= ', np.mean(self.all_batch_acc[task-1]))
      print('All batches cohen kappa= ', np.mean(self.all_batch_kappa[task-1]))
      self.acc_saving = [[] for _ in range(task+1)]
      self.cohen_kappa_saving=[[] for _ in range(task+1)]

    def plotting(self):

      x0=np.cumsum(self.all_batch_acc[0]) / np.arange(1, len(self.all_batch_acc[0]) + 1)
      x1=np.cumsum(self.all_batch_acc[1]) / np.arange(1, len(self.all_batch_acc[1]) + 1)
      x2=np.cumsum(self.all_batch_acc[2]) / np.arange(1, len(self.all_batch_acc[2]) + 1)
      x3=np.cumsum(self.all_batch_acc[3]) / np.arange(1, len(self.all_batch_acc[3]) + 1)
      all_x=np.concatenate((x0,x1,x2,x3),axis=0)
      vertical_lines_x = [len(x0), len(x0)+len(x1), len(x0)+len(x1)+len(x2)]
      y = all_x
      x = list(range(1,len(all_x) + 1))
      fig, ax = plt.subplots(figsize=(12, 4))
      ax.margins(x=0.0)
      for i in vertical_lines_x:
          plt.axvline(x=i, color='#D3D3D3', linestyle='-')
      ax.plot(x, y, color='#ff1d58')
      plt.xlabel('Batch Number')
      plt.ylabel('Cumulative Accuracy')
      plt.title('Cumulative Acuuracy Over Batches')
      plt.legend()
      plt.show()

      x0=np.cumsum(self.all_batch_kappa[0]) / np.arange(1, len(self.all_batch_kappa[0]) + 1)
      x1=np.cumsum(self.all_batch_kappa[1]) / np.arange(1, len(self.all_batch_kappa[1]) + 1)
      x2=np.cumsum(self.all_batch_kappa[2]) / np.arange(1, len(self.all_batch_kappa[2]) + 1)
      x3=np.cumsum(self.all_batch_kappa[3]) / np.arange(1, len(self.all_batch_kappa[3]) + 1)
      all_x=np.concatenate((x0,x1,x2,x3),axis=0)
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