# this file is for defining the piggyback layers with mask
import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
# importing gru math computation block
from torch._VF import gru as _VF_gru
from torch._VF import lstm as _VF_lstm

DEFAULT_THRESHOLD = 5e-3
class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):

        super(Binarizer, self).__init__()
        Binarizer.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(Binarizer.threshold)] = 0
        outputs[inputs.gt(Binarizer.threshold)] = 1
        #print(outputs)
        return outputs

    def backward(self, gradOutput):
        return gradOutput

class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):

        super(Ternarizer, self).__init__()
        Ternarizer.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > Ternarizer.threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput

def GRUBlockMath(input, hn, weight_thresholded_ih, weight_thresholded_hh, bias_ih_l0,
              bias_hh_l0, batch_size=None, bias=True, num_layers=1, dropout=0.0, training=False, bidirectional= False, batch_first=False):

  tensors = [weight_thresholded_ih,
             weight_thresholded_hh,
             bias_ih_l0,
             bias_hh_l0]
  batch_size= None
  if batch_size==None:
    output, new_hn = _VF_gru(input, hn, tensors, bias, num_layers, dropout, training, bidirectional, batch_first )
  else:
    output, new_hn = _VF_gru(input, batch_size, hn, tensors, bias, num_layers, dropout, training, bidirectional )
  return output, new_hn

def LSTMBlockMath(input, hn, weight_thresholded_ih, weight_thresholded_hh, bias_ih_l0,
              bias_hh_l0, batch_size=None, bias=True, num_layers=1, dropout=0.0, training=False, bidirectional= False, batch_first=False):

  tensors = [weight_thresholded_ih,
             weight_thresholded_hh,
             bias_ih_l0,
             bias_hh_l0]
  batch_size= None
  if batch_size==None:
    results = _VF_lstm(input, hn, tensors, bias, num_layers, dropout, training, bidirectional, batch_first )
  else:
    results = _VF_lstm(input, batch_size, hn, tensors, bias, num_layers, dropout, training, bidirectional )
  output=results[0]
  return output

class ElementWiseLSTM(nn.Module):
    """Modified linear layer."""
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
        training=False,
        bidirectional=False,
        batch_first=True,
        mask_init='uniform',
        mask_scale=1e-2,
        threshold_fn='binarizer',
      	threshold=None,
      	LSTM_weights=[],
        seq_len=10,
        LSTM_mask_weights=[]
    ):
        super(ElementWiseLSTM, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.bias=bias
        self.dropout=dropout
        self.training=training
        self.bidirectional=bidirectional
        self.batch_first=batch_first
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        self.threshold=threshold
        self.LSTM_weights=LSTM_weights
        self.seq_len=seq_len,
        self.LSTM_mask_weights=LSTM_mask_weights


        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        self.weight_ih = Variable(torch.Tensor(
            4*hidden_size, input_size), requires_grad=False)
        self.weight_hh = Variable(torch.Tensor(
            4*hidden_size, hidden_size), requires_grad=False)

        self.bias_ih_l0 = Variable(torch.Tensor(
            4*hidden_size), requires_grad=False)
        self.bias_hh_l0 = Variable(torch.Tensor(
            4*hidden_size), requires_grad=False)

        self.weight_ih=LSTM_weights[0]
        self.weight_hh=LSTM_weights[1]
        self.bias_ih_l0=LSTM_weights[2]
        self.bias_hh_l0=LSTM_weights[3]

        self.mask_real_weight_ih = self.weight_ih.data.new(self.weight_ih.size())
        self.mask_real_weight_hh = self.weight_hh.data.new(self.weight_hh.size())
        self.mask_real_bias_ih = self.weight_ih.data.new(self.bias_ih_l0.size())
        self.mask_real_bias_hh = self.weight_hh.data.new(self.bias_hh_l0.size())
        self.h0 = np.zeros((1, self.hidden_size))
        self.c0 = np.zeros((1, self.hidden_size))
        if mask_init == '1s':
            self.mask_real_weight_ih.fill_(mask_scale)
            self.mask_real_weight_hh.fill_(mask_scale)
            self.mask_real_bias_ih.fill_(mask_scale)
            self.mask_real_bias_hh.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.mask_real_weight_ih.uniform_(-1 * mask_scale, mask_scale)
            self.mask_real_weight_hh.uniform_(-1 * mask_scale, mask_scale)
            self.mask_real_bias_ih.uniform_(-1 * mask_scale, mask_scale)
            self.mask_real_bias_hh.uniform_(-1 * mask_scale, mask_scale)
        if LSTM_mask_weights!=[]:
            self.mask_real_weight_ih = Parameter(self.LSTM_mask_weights[0])
            self.mask_real_weight_hh = Parameter(self.LSTM_mask_weights[1])
            self.mask_real_bias_ih = Parameter(self.LSTM_mask_weights[2])
            self.mask_real_bias_hh = Parameter(self.LSTM_mask_weights[3])

        else:
            self.mask_real_weight_ih = Parameter(self.mask_real_weight_ih)
            self.mask_real_weight_hh = Parameter(self.mask_real_weight_hh)
            self.mask_real_bias_ih = Parameter(self.mask_real_bias_ih)
            self.mask_real_bias_hh = Parameter(self.mask_real_bias_hh)


        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self,input):
        if torch.isnan(self.mask_real_weight_ih).any():
            if self.mask_init == 'uniform':
                self.mask_real_weight_ih.uniform_(-1 * self.mask_scale, self.mask_scale)
                self.mask_real_weight_hh.uniform_(-1 * self.mask_scale, self.mask_scale)
            elif self.mask_init == '1s'
                self.mask_real_weight_ih.fill_(self.mask_scale)
                self.mask_real_weight_hh.fill_(self.mask_scale)

            self.mask_real_weight_ih = Parameter(self.mask_real_weight_ih)
            self.mask_real_weight_hh = Parameter(self.mask_real_weight_hh)


        mask_thresholded_ih = self.threshold_fn.apply(self.mask_real_weight_ih)
        mask_thresholded_hh = self.threshold_fn.apply(self.mask_real_weight_hh)
        mask_thresholded_bias_ih = self.threshold_fn.apply(self.mask_real_bias_ih)
        mask_thresholded_bias_hh = self.threshold_fn.apply(self.mask_real_bias_hh)
        self.hn=self._build_initial_state(input, self.h0)
        self.cn=self._build_initial_state(input, self.c0)
        self.hx=(self.hn,self.cn)

        weight_thresholded_ih = mask_thresholded_ih * self.weight_ih
        weight_thresholded_hh = mask_thresholded_hh * self.weight_hh
        weight_thresholded_bias_ih = mask_thresholded_bias_ih * self.bias_ih_l0
        weight_thresholded_bias_hh = mask_thresholded_bias_hh * self.bias_hh_l0

        out = LSTMBlockMath(input, self.hx, weight_thresholded_ih, weight_thresholded_hh,
                                weight_thresholded_bias_ih, weight_thresholded_bias_hh, self.batch_size, self.bias, self.num_layers, self.dropout,
                                self.training, self.bidirectional, self.batch_first)

        return out
    def _build_initial_state(self, x, state):
        s = torch.from_numpy(np.tile(state, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight_ih.data = fn(self.weight_ih.data)
        self.weight_hh.data = fn(self.weight_hh.data)
        self.bias_ih_l0.data = fn(self.bias_ih_l0.data)
        self.bias_hh_l0.data = fn(self.bias_ih_l0.data)
        #if self.bias is not None and self.bias_ih_l0.data is not None:
        #    self.bias_ih_l0.data = fn(self.bias_ih_l0.data)
        #if self.bias is not None and self.bias_hh_l0.data is not None:
        #    self.bias_hh_l0.data = fn(self.bias_hh_l0.data)


class ElementWiseGRU(nn.Module):
    """Modified linear layer."""
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
        training=False,
        bidirectional=False,
        batch_first=True,
        mask_init='uniform',
        mask_scale=1e-2,
        threshold_fn='binarizer',
      	threshold=None,
      	GRU_weights=[],
        seq_len=10,
        GRU_mask_weights=[]
    ):
        super(ElementWiseGRU, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.bias=bias
        self.dropout=dropout
        self.training=training
        self.bidirectional=bidirectional
        self.batch_first=batch_first
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        self.threshold=threshold
        self.GRU_weights=GRU_weights
        self.seq_len=seq_len,
        self.GRU_mask_weights=GRU_mask_weights

        self.h0 = np.zeros((1, self.hidden_size))

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        self.weight_ih = Variable(torch.Tensor(
            3*hidden_size, input_size), requires_grad=False)
        self.weight_hh = Variable(torch.Tensor(
            3*hidden_size, hidden_size), requires_grad=False)

        self.bias_ih_l0 = Variable(torch.Tensor(
            3*hidden_size), requires_grad=False)
        self.bias_hh_l0 = Variable(torch.Tensor(
            3*hidden_size), requires_grad=False)

        self.weight_ih=GRU_weights[0]
        self.weight_hh=GRU_weights[1]
        self.bias_ih_l0=GRU_weights[2]
        self.bias_hh_l0=GRU_weights[3]

        self.mask_real_weight_ih = self.weight_ih.data.new(self.weight_ih.size())
        self.mask_real_weight_hh = self.weight_hh.data.new(self.weight_hh.size())
        self.mask_real_bias_ih = self.weight_ih.data.new(self.bias_ih_l0.size())
        self.mask_real_bias_hh = self.weight_hh.data.new(self.bias_hh_l0.size())
        if mask_init == '1s':
            self.mask_real_weight_ih.fill_(mask_scale)
            self.mask_real_weight_hh.fill_(mask_scale)
            self.mask_real_bias_ih.fill_(mask_scale)
            self.mask_real_bias_hh.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.mask_real_weight_ih.uniform_(-1 * mask_scale, mask_scale)
            self.mask_real_weight_hh.uniform_(-1 * mask_scale, mask_scale)
            self.mask_real_bias_ih.uniform_(-1 * mask_scale, mask_scale)
            self.mask_real_bias_hh.uniform_(-1 * mask_scale, mask_scale)
        if GRU_mask_weights!=[]:
            self.mask_real_weight_ih = Parameter(self.GRU_mask_weights[0])
            self.mask_real_weight_hh = Parameter(self.GRU_mask_weights[1])
            self.mask_real_bias_ih = Parameter(self.GRU_mask_weights[2])
            self.mask_real_bias_hh = Parameter(self.GRU_mask_weights[3])

        else:
            self.mask_real_weight_ih = Parameter(self.mask_real_weight_ih)
            self.mask_real_weight_hh = Parameter(self.mask_real_weight_hh)
            self.mask_real_bias_ih = Parameter(self.mask_real_bias_ih)
            self.mask_real_bias_hh = Parameter(self.mask_real_bias_hh)


        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self,input):
        if torch.isnan(self.mask_real_weight_ih).any():
            if self.mask_init == 'uniform':
                self.mask_real_weight_ih.uniform_(-1 * self.mask_scale, self.mask_scale)
                self.mask_real_weight_hh.uniform_(-1 * self.mask_scale, self.mask_scale)
            if self.mask_init == '1s':
                self.mask_real_weight_ih.fill_(self.mask_scale)
                self.mask_real_weight_hh.fill_(self.mask_scale)
            self.mask_real_weight_ih = Parameter(self.mask_real_weight_ih)
            self.mask_real_weight_hh = Parameter(self.mask_real_weight_hh)

        
        self.hn=self._build_initial_state(input, self.h0)
        mask_thresholded_ih = self.threshold_fn.apply(self.mask_real_weight_ih)
        mask_thresholded_hh = self.threshold_fn.apply(self.mask_real_weight_hh)
        mask_thresholded_bias_ih = self.threshold_fn.apply(self.mask_real_bias_ih)
        mask_thresholded_bias_hh = self.threshold_fn.apply(self.mask_real_bias_hh)


        weight_thresholded_ih = mask_thresholded_ih * self.weight_ih
        weight_thresholded_hh = mask_thresholded_hh * self.weight_hh
        weight_thresholded_bias_ih = mask_thresholded_bias_ih * self.bias_ih_l0
        weight_thresholded_bias_hh = mask_thresholded_bias_hh * self.bias_hh_l0

        out,_ = GRUBlockMath(input, self.hn, weight_thresholded_ih, weight_thresholded_hh,
                                weight_thresholded_bias_ih, weight_thresholded_bias_hh, self.batch_size, self.bias, self.num_layers, self.dropout,
                                self.training, self.bidirectional, self.batch_first)


        return out
    def _build_initial_state(self, x, state):
        s = torch.from_numpy(np.tile(state, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight_ih.data = fn(self.weight_ih.data)
        self.weight_hh.data = fn(self.weight_hh.data)
        self.bias_ih_l0.data = fn(self.bias_ih_l0.data)
        self.bias_hh_l0.data = fn(self.bias_ih_l0.data)
        #if self.bias is not None and self.bias_ih_l0.data is not None:
        #    self.bias_ih_l0.data = fn(self.bias_ih_l0.data)
        #if self.bias is not None and self.bias_hh_l0.data is not None:
        #    self.bias_hh_l0.data = fn(self.bias_hh_l0.data)


class ElementWiseLinear(nn.Module):
    """Modified linear layer."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mask_init='uniform',
        mask_scale=1e-2,
        threshold_fn='binarizer',
        threshold=None,
        linear_weights=[],
        Linear_mask_weights=[]
        ):
        super(ElementWiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.linear_weights=linear_weights
        self.Linear_mask_weights=Linear_mask_weights
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }


        self.weight = Variable(torch.Tensor(
            out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(
                out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.weight=linear_weights[0]
        self.bias=linear_weights[1]

        self.mask_real_linear = self.weight.data.new(self.weight.size())
        if mask_init == '1s':
            self.mask_real_linear.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.mask_real_linear.uniform_(-1 * mask_scale, mask_scale)

        if Linear_mask_weights!=[]:
            self.mask_real_linear = Parameter(self.Linear_mask_weights)
        else:
            self.mask_real_linear = Parameter(self.mask_real_linear)

        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input):

        mask_thresholded = self.threshold_fn.apply(self.mask_real_linear)

        weight_thresholded = mask_thresholded * self.weight

        return F.linear(input, weight_thresholded, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)