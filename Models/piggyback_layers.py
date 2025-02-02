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
                 bias_hh_l0, batch_size=None, bias=True, num_layers=1, dropout=0.0,
                 training=False, bidirectional=False, batch_first=True, sample_wise=False):
    """
    GRU block computation using thresholded weights and optional sample-wise processing.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor of shape (batch_size, seq_len, input_size).
    hn : torch.Tensor
        The initial hidden state tensor.
    weight_thresholded_ih : torch.Tensor
        The input-hidden weight tensor after thresholding.
    weight_thresholded_hh : torch.Tensor
        The hidden-hidden weight tensor after thresholding.
    bias_ih_l0 : torch.Tensor
        The bias tensor for input-hidden weights.
    bias_hh_l0 : torch.Tensor
        The bias tensor for hidden-hidden weights.
    batch_size : int, optional
        The batch size of the input (default is None).
    bias : bool, optional
        Whether to use biases in the GRU computation (default is True).
    num_layers : int, optional
        Number of stacked GRU layers (default is 1).
    dropout : float, optional
        Dropout rate between GRU layers (default is 0.0).
    training : bool, optional
        Whether the model is in training mode (default is False).
    bidirectional : bool, optional
        Whether the GRU is bidirectional (default is False).
    batch_first : bool, optional
        Whether the input tensor has batch size as the first dimension (default is False).
    sample_wise : bool, optional
        Whether to process input samples sequentially, updating hidden states iteratively (default is False).

    Returns
    -------
    output : torch.Tensor
        The output tensor from the GRU layer.
    new_hn : torch.Tensor
        The updated hidden state tensor.
    """
    tensors = [weight_thresholded_ih, weight_thresholded_hh, bias_ih_l0, bias_hh_l0]
    batch_size= None
    if batch_size is None:
        # print(input.size())
        # print(hn.size())
        # print('bidirectional',bidirectional)
        # print('training',training)
        # print('dropout,',dropout)
        # print("Weight IH shape:", weight_thresholded_ih.shape)
        # print("Weight HH shape:", weight_thresholded_hh.shape)
        # print("Bias IH shape:", bias_ih_l0.shape)
        # print("Bias HH shape:", bias_hh_l0.shape)
        # print('batch_first',batch_first)
        output, new_hn = _VF_gru(input, hn, tensors, bias, num_layers, dropout, training,
                                 bidirectional, batch_first)
    elif sample_wise:
        outputs = []
        for i in range(input.size(0)):  # Process each sample independently
            sample_output, hn = _VF_gru(input[i].unsqueeze(0), hn, tensors, bias, num_layers,
                                        dropout, training, bidirectional, batch_first)
            outputs.append(sample_output)
        output = torch.cat(outputs, dim=0)  # Concatenate outputs to form the final tensor
        new_hn = hn
    else:
        batch_size = torch.tensor(batch_size)
        output, new_hn = _VF_gru(input, batch_size, hn, tensors, bias, num_layers, dropout,
                                 training, bidirectional)

    return output, new_hn

def LSTMBlockMath(input, hn, weight_thresholded_ih, weight_thresholded_hh, bias_ih_l0,
                  bias_hh_l0, batch_size=None, bias=True, num_layers=1, dropout=0.0,
                  training=False, bidirectional=False, batch_first=True, sample_wise=False):
    """
    LSTM block computation using thresholded weights and optional sample-wise processing.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor of shape (batch_size, seq_len, input_size).
    hn : tuple of torch.Tensor
        The initial hidden state (h, c) for the LSTM.
    weight_thresholded_ih : torch.Tensor
        The input-hidden weight tensor after thresholding.
    weight_thresholded_hh : torch.Tensor
        The hidden-hidden weight tensor after thresholding.
    bias_ih_l0 : torch.Tensor
        The bias tensor for input-hidden weights.
    bias_hh_l0 : torch.Tensor
        The bias tensor for hidden-hidden weights.
    batch_size : int, optional
        The batch size of the input (default is None).
    bias : bool, optional
        Whether to use biases in the LSTM computation (default is True).
    num_layers : int, optional
        Number of stacked LSTM layers (default is 1).
    dropout : float, optional
        Dropout rate between LSTM layers (default is 0.0).
    training : bool, optional
        Whether the model is in training mode (default is False).
    bidirectional : bool, optional
        Whether the LSTM is bidirectional (default is False).
    batch_first : bool, optional
        Whether the input tensor has batch size as the first dimension (default is False).
    sample_wise : bool, optional
        Whether to process input samples sequentially, updating hidden states iteratively (default is False).

    Returns
    -------
    output : torch.Tensor
        The output tensor from the LSTM layer.
    hidden : tuple of torch.Tensor
        The updated hidden state tensor (h, c).
    """
    tensors = [weight_thresholded_ih, weight_thresholded_hh, bias_ih_l0, bias_hh_l0]

    if batch_size is None:
        results = _VF_lstm(input, hn, tensors, bias, num_layers, dropout, training,
                           bidirectional, batch_first)
    elif sample_wise:
        outputs = []
        for i in range(input.size(0)):  # Process each sample independently
            sample_result = _VF_lstm(input[i].unsqueeze(0), hn, tensors, bias, num_layers,
                                     dropout, training, bidirectional, batch_first)
            outputs.append(sample_result[0])  # Collecting output of each sample
            hn = sample_result[1:]  # Update hidden state with the new one
        output = torch.cat(outputs, dim=0)  # Concatenate outputs to form the final tensor
        hidden = hn
    else:
        results = _VF_lstm(input, batch_size, hn, tensors, bias, num_layers, dropout,
                           training, bidirectional)

    # Extract output and hidden state
    output = results[0]
    hidden = results[1:]

    return output, hidden

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
        # this variable should be asked from TA
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

        # PARAMETERS
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

        # this hn should be defined at the begining and will be updated during the training.
        # I should ask from TA about configuration of hn. if it needed to be updated after each batch or it initialized to zero each iteration
        #self.h0 = torch.randn((num_layers, seq_len, self.hidden_size))
        #self.c0 = torch.rand((num_layers, seq_len, self.hidden_size))
        #hx = (self.hn, self.c0)

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        # weight and bias are no longer Parameters.
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
        # Initialize real-valued mask weights.
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
            print('NaN exists ............................................')
            #print(self.mask_real_weight_ih)
            if self.mask_init == 'uniform':
                self.mask_real_weight_ih.uniform_(-1 * self.mask_scale, self.mask_scale)
                self.mask_real_weight_hh.uniform_(-1 * self.mask_scale, self.mask_scale)
            self.mask_real_weight_ih = Parameter(self.mask_real_weight_ih)
            self.mask_real_weight_hh = Parameter(self.mask_real_weight_hh)

        # Get binarized/ternarized mask from real-valued mask.

        mask_thresholded_ih = self.threshold_fn.apply(self.mask_real_weight_ih)
        mask_thresholded_hh = self.threshold_fn.apply(self.mask_real_weight_hh)
        mask_thresholded_bias_ih = self.threshold_fn.apply(self.mask_real_bias_ih)
        mask_thresholded_bias_hh = self.threshold_fn.apply(self.mask_real_bias_hh)
        self.hn=self._build_initial_state(input, self.h0)
        self.cn=self._build_initial_state(input, self.c0)
        self.hx=(self.hn,self.cn)
        # Mask weights with above mask.
        weight_thresholded_ih = mask_thresholded_ih * self.weight_ih
        weight_thresholded_hh = mask_thresholded_hh * self.weight_hh
        weight_thresholded_bias_ih = mask_thresholded_bias_ih * self.bias_ih_l0
        weight_thresholded_bias_hh = mask_thresholded_bias_hh * self.bias_hh_l0

        out = LSTMBlockMath(input, self.hx, weight_thresholded_ih, weight_thresholded_hh,
                                weight_thresholded_bias_ih, weight_thresholded_bias_hh, self.batch_size, self.bias, self.num_layers, self.dropout,
                                self.training, self.bidirectional, self.batch_first)
        # Get output using modified weight.

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
    """Modified GRU layer with mask-based pruning, similar initialization to the first class."""
    
    def __init__(
        self,
        input_size=2,
        device=torch.device("cpu"),
        num_layers=1,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        many_to_one=False,
        remember_states=None,
        bias=True,
        dropout=0.0,
        training=False,
        bidirectional=False,
        batch_first=True,
        mask_init="uniform",
        mask_scale=1e-2,
        threshold_fn="binarizer",
        threshold=None,
        GRU_weights=[],
        seq_len=10,
        GRU_mask_weights=[],
        model_type="CPB",
        mask_option="SUM",
        low_rank=False,  # Enable low-rank decomposition
        rank_dim=10,  # Rank for low-rank approximation
        weight_init=None,
        sample_wise=False
    ):
        super(ElementWiseGRU, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.bias = bias
        self.dropout = dropout
        self.training = training
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        self.threshold = threshold
        self.GRU_weights = GRU_weights
        self.seq_len = seq_len
        self.GRU_mask_weights = GRU_mask_weights
        self.many_to_one = many_to_one
        self.remember_states = remember_states
        self.low_rank = low_rank
        self.rank_dim = rank_dim  # Rank dimension for low-rank decomposition
        self.weight_init = weight_init
        self.mask_option = mask_option
        self.sample_wise = sample_wise

        # INITIALIZATION OF WEIGHTS AND MASKS
        # Match the initialization of weights and masks to the first class

        # Initialize weights and biases
        self.weight_ih = Variable(torch.Tensor(3 * hidden_size, input_size), requires_grad=False)
        self.weight_hh = Variable(torch.Tensor(3 * hidden_size, hidden_size), requires_grad=False)
        self.bias_ih_l0 = Variable(torch.Tensor(3 * hidden_size), requires_grad=False)
        self.bias_hh_l0 = Variable(torch.Tensor(3 * hidden_size), requires_grad=False)

        # Assign the provided weights
        self.weight_ih = GRU_weights[0]
        self.weight_hh = GRU_weights[1]
        self.bias_ih_l0 = GRU_weights[2]
        self.bias_hh_l0 = GRU_weights[3]

        # Initialize mask weights
        self.mask_real_weight_ih = self.weight_ih.data.new(self.weight_ih.size())
        self.mask_real_weight_hh = self.weight_hh.data.new(self.weight_hh.size())
        self.mask_real_bias_ih = self.weight_ih.data.new(self.bias_ih_l0.size())
        self.mask_real_bias_hh = self.weight_hh.data.new(self.bias_hh_l0.size())

        if mask_init == "1s":
            # Fill masks with ones scaled by mask_scale
            self.mask_real_weight_ih.fill_(mask_scale)
            self.mask_real_weight_hh.fill_(mask_scale)
            self.mask_real_bias_ih.fill_(mask_scale)
            self.mask_real_bias_hh.fill_(mask_scale)
        elif mask_init == "uniform":
            # Uniform initialization of masks
            self.mask_real_weight_ih.uniform_(-mask_scale, mask_scale)
            self.mask_real_weight_hh.uniform_(-mask_scale, mask_scale)
            self.mask_real_bias_ih.uniform_(-mask_scale, mask_scale)
            self.mask_real_bias_hh.uniform_(-mask_scale, mask_scale)

        # Assign provided mask weights if available
        if GRU_mask_weights != []:
            self.mask_real_weight_ih = Parameter(GRU_mask_weights[0])
            self.mask_real_weight_hh = Parameter(GRU_mask_weights[1])
            self.mask_real_bias_ih = Parameter(GRU_mask_weights[2])
            self.mask_real_bias_hh = Parameter(GRU_mask_weights[3])
        else:
            # Make the masks trainable parameters
            self.mask_real_weight_ih = Parameter(self.mask_real_weight_ih)
            self.mask_real_weight_hh = Parameter(self.mask_real_weight_hh)
            self.mask_real_bias_ih = Parameter(self.mask_real_bias_ih)
            self.mask_real_bias_hh = Parameter(self.mask_real_bias_hh)

        # Initialize the thresholder function
        if threshold_fn == "binarizer":
            self.threshold_fn = Binarizer(threshold=self.threshold)
        elif threshold_fn == "ternarizer":
            self.threshold_fn = Ternarizer(threshold=self.threshold)

        # Initialize the initial hidden state
        self.h0 = np.zeros((1, self.hidden_size))

    def forward(self, input, prev_h=None, train=True):
        # Handle NaN values in the mask
        if torch.isnan(self.mask_real_weight_ih).any():
            print('NaN detected in mask weights. Reinitializing masks...')
            if self.mask_init == 'uniform':
                self.mask_real_weight_ih.uniform_(-1 * self.mask_scale, self.mask_scale)
                self.mask_real_weight_hh.uniform_(-1 * self.mask_scale, self.mask_scale)
            # Make sure masks are trainable parameters
            self.mask_real_weight_ih = Parameter(self.mask_real_weight_ih)
            self.mask_real_weight_hh = Parameter(self.mask_real_weight_hh)

        # Concatenate previous hidden state if provided
        if prev_h is not None:
            prev_h = prev_h.to(self.device)
            input = torch.cat((input, prev_h), dim=2)  # (B, L, I+H)

        # Apply the mask based on the chosen method
        if self.mask_option == "DOT":
            # Threshold-based mask application
            mask_thresholded_ih = self.threshold_fn.apply(self.mask_real_weight_ih)
            mask_thresholded_hh = self.threshold_fn.apply(self.mask_real_weight_hh)
            mask_thresholded_bias_ih = self.threshold_fn.apply(self.mask_real_bias_ih)
            mask_thresholded_bias_hh = self.threshold_fn.apply(self.mask_real_bias_hh)

            # Apply the mask element-wise
            weight_thresholded_ih = mask_thresholded_ih * self.weight_ih
            weight_thresholded_hh = mask_thresholded_hh * self.weight_hh
            weight_thresholded_bias_ih = mask_thresholded_bias_ih * self.bias_ih_l0
            weight_thresholded_bias_hh = mask_thresholded_bias_hh * self.bias_hh_l0
        else:
            # Sum-based mask application
            weight_thresholded_ih = self.weight_ih + self.mask_real_weight_ih
            weight_thresholded_hh = self.weight_hh + self.mask_real_weight_hh
            weight_thresholded_bias_ih = self.bias_ih_l0 + self.mask_real_bias_ih
            weight_thresholded_bias_hh = self.bias_hh_l0 + self.mask_real_bias_hh

        # Build initial hidden state
        self.hn = self._build_initial_state(input, self.h0)

        # GRU block math with masked weights
        out, hn = GRUBlockMath(
            input, self.hn, weight_thresholded_ih, weight_thresholded_hh,
            weight_thresholded_bias_ih, weight_thresholded_bias_hh, self.batch_size,
            self.bias, self.num_layers, self.dropout, self.training,
            self.bidirectional, self.batch_first, self.sample_wise
        )

        # If many-to-one mode, take the last output
        if self.many_to_one:
            out = out[:, -1, :]

        # Update the hidden state if `remember_states` is True
        if train and self.remember_states:
            self.h0 = hn.detach().numpy()

        return out

    def _build_initial_state(self, x, state):
        # Build the initial hidden state
        s = torch.from_numpy(np.tile(state, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)

    def _apply(self, fn):
        # Apply a function to all parameters and buffers
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight_ih.data = fn(self.weight_ih.data)
        self.weight_hh.data = fn(self.weight_hh.data)
        self.bias_ih_l0.data = fn(self.bias_ih_l0.data)
        self.bias_hh_l0.data = fn(self.bias_hh_l0.data)


class ElementWiseLinear(nn.Module):
    """Modified linear layer with mask-based pruning and optional low-rank approximation."""
    
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mask_init="uniform",
        mask_scale=1e-2,
        threshold_fn="binarizer",
        threshold=None,
        linear_weights=[],
        Linear_mask_weights=[],
        model_type="CPB",
        mask_option="DOT",
        low_rank=False,  # Enable low-rank decomposition
        rank_dim=10,  # Rank for low-rank approximation
        weight_init=None
    ):
        super(ElementWiseLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.linear_weights = linear_weights
        self.Linear_mask_weights = Linear_mask_weights
        self.model_type = model_type
        self.mask_option = mask_option
        self.low_rank = low_rank
        self.rank_dim = rank_dim  # Rank dimension for low-rank decomposition
        self.weight_init = weight_init

        if threshold is None:
            threshold = 5e-3  # Default threshold
        
        self.info = {
            "threshold_fn": threshold_fn,
            "threshold": threshold,
        }

        self.weight = Variable(torch.Tensor(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        
        self.weight = linear_weights[0]
        self.bias = linear_weights[1]

        if low_rank:
            # Initialize low-rank matrices
            self.mask_H = Parameter(torch.randn(out_features, rank_dim) * mask_scale)
            self.mask_V = Parameter(torch.randn(rank_dim, in_features) * mask_scale)
        else:
            self.mask_real_linear = self.weight.data.new(self.weight.size())
            if mask_init == "1s":
                self.mask_real_linear.fill_(mask_scale)
            elif mask_init == "uniform":
                self.mask_real_linear.uniform_(-1 * mask_scale, mask_scale)

            if Linear_mask_weights != []:
                self.mask_real_linear = Parameter(self.Linear_mask_weights)
            else:
                self.mask_real_linear = Parameter(self.mask_real_linear)

        if self.threshold_fn == "binarizer":
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == "ternarizer":
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input):
        if self.low_rank:
            # Compute low-rank mask
            mask_real_linear = torch.matmul(self.mask_H, self.mask_V)
        else:
            mask_real_linear = self.mask_real_linear

        if self.mask_option == "SUM":
            # Sum-based mask application
            weight_summed = self.weight + mask_real_linear
            return F.linear(input, weight_summed, self.bias)
        else:
            # Threshold-based mask application
            mask_thresholded = self.threshold_fn.apply(mask_real_linear)
            weight_thresholded = mask_thresholded * self.weight
            return F.linear(input, weight_thresholded, self.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features})"

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if self.bias is not None:
            self.bias.data = fn(self.bias.data)