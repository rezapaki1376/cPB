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
from models.clstm import (
    cLSTMLinear,
)
from models.cgru import (
    cGRULinear,
)
from models.PiggyBackGRU import(
	PiggyBackGRU,
)
from models.network import ModifiedGRU

class cPNN:
    """
    Class that implements all the cPNN structure.
    """

    def __init__(
        self,
        model_class=PiggyBackGRU,
        device=None,
        stride: int = 1,
        lr: float = 0.01,
        seq_len: int = 5,
        pretrain_model_addr='',
        mask_weights=[],
        mask_init='1s',
        number_of_tasks=4,
        **kwargs,
    ):
      
      self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
      self.stride=stride
      self.seq_len=seq_len
      self.lr=lr
      self.pretrain_model_addr=pretrain_model_addr
      self.mask_init=mask_init
      
      if model_class==PiggyBackGRU and pretrain_model_addr!='':
        self.model = ModifiedGRU(pretrain_model_addr=pretrain_model_addr,seq_len=seq_len,mask_weights=mask_weights,mask_init=mask_init)
        self.initial_weights = self.model.state_dict()
        
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


    def learn_many(self,x,y,weights):
      if weights!=[]:
        self.model=self.add_new_column(weights)
      else:
        self.model=self.model
      x = np.array(x)
      y = list(y)

      optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
      x, y, _ = self._load_batch(x, y)
      y_pred = self.model(x)
      y_pred = get_samples_outputs(y_pred)
      pred, _ = get_pred_from_outputs(y_pred)
      loss = self.loss_fn(y_pred, y)

      # backward pass

      optimizer.zero_grad()

      loss.backward(retain_graph=True)
      # update weights
      optimizer.step()
      kappa=cohen_kappa(y,pred).item()
      weights = self.model.state_dict()
      acc = accuracy_score(np.array(y),np.array(pred))
      return weights,acc,kappa


    def predict_many(self,x,y):
      x = np.array(x)
      y = list(y)
      x, y, _ = self._load_batch(x, y)
      y_pred = self.model(x)
      y_pred = get_samples_outputs(y_pred)
      pred, _ = get_pred_from_outputs(y_pred)
      return accuracy_score(np.array(y),np.array(pred))

    def initial_weights_returning(self):
      return self.initial_weights
    def add_new_column(self,new_mask):
      param_list=[]
      for params in new_mask:
        param_list.append(params)
      
      mask_weights=[]
      mask_weights.append(new_mask[param_list[-5]])
      mask_weights.append(new_mask[param_list[-4]])
      mask_weights.append(new_mask[param_list[-3]])
      mask_weights.append(new_mask[param_list[-2]])
      mask_weights.append(new_mask[param_list[-1]])
      self.model=ModifiedGRU(pretrain_model_addr=self.pretrain_model_addr,
                             seq_len=self.seq_len,mask_weights=mask_weights,
                             mask_init=self.mask_init)
      return self.model