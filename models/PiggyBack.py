# this file is for Main network that contains piggyback layers with masks.
# here we just
import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import models.piggyback_layers as nl


class PiggyBackGRU(nn.Module):
    def __init__(
        self,
        input_size=2,
        device=torch.device("cpu"),
      	num_layers=1,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        many_to_one=False,
        remember_states = None,
        bias=True,
        dropout=0.0,
        # this variable should be asked from TA
        training=False,
        bidirectional=False,
        batch_first=False,
        mask_init='uniform',
      	mask_scale=1e-2,
        threshold_fn='binarizer',
      	threshold=None,
      	all_weights=[],
        seq_len=10,
        mask_weights=[]
        ):
        super(PiggyBackGRU, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.all_weights=all_weights
        self.seq_len=seq_len
        self.mask_weights=mask_weights

        self.gru_weight_ih_l0=all_weights['gru.weight_ih_l0']
        self.gru_weight_hh_l0=all_weights['gru.weight_hh_l0']
        self.gru_bias_ih_l0=all_weights['gru.bias_ih_l0']
        self.gru_bias_hh_l0=all_weights['gru.bias_hh_l0']
        self.linear_weight=all_weights['linear.weight']
        self.linear_bias=all_weights['linear.bias']
        self.GRU_weights=[self.gru_weight_ih_l0,
           		   self.gru_weight_hh_l0,
           		   self.gru_bias_ih_l0,
           		   self.gru_bias_hh_l0]

        self.linear_weights=[
          			self.linear_weight,
          			self.linear_bias]
        if mask_weights!=[]:
            self.GRU_mask_weights=mask_weights[0:4]
            self.Linear_mask_weights=mask_weights[-1]
        else:
            self.GRU_mask_weights=[]
            self.Linear_mask_weights=[]
        # define nn network here

        self.classifier = nn.Sequential(
            nl.ElementWiseGRU(input_size=input_size, device=device, num_layers=num_layers, hidden_size=hidden_size, bias=bias, dropout=dropout,
                bidirectional=bidirectional, training=training, mask_init=mask_init,
                mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold,
                GRU_weights=self.GRU_weights, seq_len=self.seq_len, GRU_mask_weights=self.GRU_mask_weights),

            nl.ElementWiseLinear(in_features=hidden_size, out_features=output_size,
                mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,
				threshold=threshold, linear_weights=self.linear_weights, Linear_mask_weights=self.Linear_mask_weights)
        )


    def forward(self,input):
        out = self.classifier(input)
        return out



class PiggyBackLSTM(nn.Module):
    def __init__(
        self,
        input_size=2,
        device=torch.device("cpu"),
      	num_layers=1,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        many_to_one=False,
        remember_states = None,
        bias=True,
        dropout=0.0,
        # this variable should be asked from TA
        training=False,
        bidirectional=False,
        batch_first=False,
        mask_init='uniform',
      	mask_scale=1e-2,
        threshold_fn='binarizer',
      	threshold=None,
      	all_weights=[],
        seq_len=10,
        mask_weights=[]
        ):
        super(PiggyBackLSTM, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.all_weights=all_weights
        self.seq_len=seq_len
        self.mask_weights=mask_weights

        self.lstm_weight_ih_l0=all_weights['lstm.weight_ih_l0']
        self.lstm_weight_hh_l0=all_weights['lstm.weight_hh_l0']
        self.lstm_bias_ih_l0=all_weights['lstm.bias_ih_l0']
        self.lstm_bias_hh_l0=all_weights['lstm.bias_hh_l0']
        self.linear_weight=all_weights['linear.weight']
        self.linear_bias=all_weights['linear.bias']
        self.LSTM_weights=[self.lstm_weight_ih_l0,
           		   self.lstm_weight_hh_l0,
           		   self.lstm_bias_ih_l0,
           		   self.lstm_bias_hh_l0]

        self.linear_weights=[
          			self.linear_weight,
          			self.linear_bias]
        if mask_weights!=[]:
            self.LSTM_mask_weights=mask_weights[0:4]
            self.Linear_mask_weights=mask_weights[-1]
        else:
            self.LSTM_mask_weights=[]
            self.Linear_mask_weights=[]
        # define nn network here

        self.classifier = nn.Sequential(
            nl.ElementWiseLSTM(input_size=input_size, device=device, num_layers=num_layers, hidden_size=hidden_size, bias=bias, dropout=dropout,
                bidirectional=bidirectional, training=training, mask_init=mask_init,
                mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold,
                LSTM_weights=self.LSTM_weights, seq_len=self.seq_len, LSTM_mask_weights=self.LSTM_mask_weights),

            nl.ElementWiseLinear(in_features=hidden_size, out_features=output_size,
                mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,
				threshold=threshold, linear_weights=self.linear_weights, Linear_mask_weights=self.Linear_mask_weights)
        )


    def forward(self,input):
        out = self.classifier(input)
        return out