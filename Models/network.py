import torch
import numpy as np
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
#from models.pretrain import *
from models.PiggyBack import *
import pickle
class ModifiedRNN(nn.Module):
    def __init__(
        self,
        input_size=2,
        device=torch.device("cpu"),
      	num_layers=1,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        base_model='gru',
        many_to_one=False,
        remember_states = None,
        bias=True,
        dropout=0.0,
        # this variable should be asked from TA
        training=False,
        bidirectional=False,
        batch_first=False,
        mask_init='unifrom',
      	mask_scale=1e-2,
        threshold_fn='binarizer',
      	threshold=None,
      	pretrain_model_addr='',
        seq_len=10,
        mask_weights=[]
    ):
        super(ModifiedRNN, self).__init__()


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
        self.pretrain_model_addr=pretrain_model_addr
        self.seq_len=seq_len
        self.mask_weights=mask_weights
        self.base_model=base_model
        if base_model == 'gru':
            self.pretrain_model = GRU_Model(
            input_size=input_size,
            device=torch.device("cpu"),
      	    num_layers=num_layers,
            hidden_size=hidden_size,
            output_size=output_size,
            batch_size=batch_size,
              )
        else:
            self.pretrain_model = LSTM_Model(
            input_size=input_size,
            device=torch.device("cpu"),
      	    num_layers=num_layers,
            hidden_size=hidden_size,
            output_size=output_size,
            batch_size=batch_size,
              )

        with open(self.pretrain_model_addr, "rb") as fp:
    		    self.pretrain_model.load_state_dict(pickle.load(fp),strict=False)
        self.all_weights=self.pretrain_model.state_dict()
        if base_model =='gru':
            self.classifier=cPBGRU(input_size=input_size, device=device,
                               num_layers=num_layers, hidden_size=hidden_size,
                               output_size=output_size, batch_size=batch_size,
                               many_to_one=many_to_one,remember_states=remember_states,
                               bias=bias, training=training, dropout=dropout,
                               bidirectional=bidirectional, batch_first=batch_first,
                               mask_init=mask_init, mask_scale=mask_scale,
                               threshold_fn=threshold_fn, threshold=threshold,
                               all_weights=self.all_weights,seq_len=seq_len, mask_weights=mask_weights
                               )
        else:
            self.classifier=cPBLSTM(input_size=input_size, device=device,
                               num_layers=num_layers, hidden_size=hidden_size,
                               output_size=output_size, batch_size=batch_size,
                               many_to_one=many_to_one,remember_states=remember_states,
                               bias=bias, training=training, dropout=dropout,
                               bidirectional=bidirectional, batch_first=batch_first,
                               mask_init=mask_init, mask_scale=mask_scale,
                               threshold_fn=threshold_fn, threshold=threshold,
                               all_weights=self.all_weights,seq_len=seq_len, mask_weights=mask_weights
                               )


    def forward(self,input):
        out = self.classifier(input)
        return out