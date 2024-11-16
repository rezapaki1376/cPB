import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import Models.piggyback_layers as nl


class PBGRU(nn.Module):
    """
    A piggyback-enabled GRU (Gated Recurrent Unit) model with element-wise masking for
    fine-grained weight adjustments.

    Parameters
    ----------
    input_size : int, default=2
        The number of input features.
    device : torch.device, default=torch.device("cpu")
        The device to run the model on ('cpu' or 'cuda').
    num_layers : int, default=1
        The number of stacked GRU layers.
    hidden_size : int, default=50
        The number of features in the GRU hidden state.
    output_size : int, default=2
        The number of output features.
    batch_size : int, default=128
        The number of samples per batch.
    many_to_one : bool, default=False
        Whether the GRU predicts only the last output (many-to-one).
    remember_states : bool or None, default=None
        Whether to remember hidden states across forward passes.
    bias : bool, default=True
        Whether to include a bias term in the GRU layers.
    dropout : float, default=0.0
        Dropout probability for the GRU layers.
    training : bool, default=False
        Whether the model is in training mode.
    bidirectional : bool, default=False
        Whether the GRU layers are bidirectional.
    batch_first : bool, default=False
        Whether the input tensors have batch size as the first dimension.
    mask_init : str, default='uniform'
        The initialization method for piggyback masks.
    mask_scale : float, default=1e-2
        The scaling factor for mask initialization.
    threshold_fn : str, default='binarizer'
        The function used to threshold the mask values.
    threshold : float or None, default=None
        The threshold value for the mask.
    all_weights : list, default=[]
        The pretrained weights for the GRU and linear layers.
    seq_len : int, default=10
        The sequence length for input data.
    mask_weights : list, default=[]
        The weights used for masking in the piggyback model.

    Attributes
    ----------
    GRU_weights : list
        The GRU weights and biases loaded from pretrained weights.
    linear_weights : list
        The linear layer weights and biases loaded from pretrained weights.
    GRU_mask_weights : list
        The mask weights for the GRU layers.
    Linear_mask_weights : list
        The mask weights for the linear layer.
    classifier : nn.Sequential
        The sequential model combining the piggyback-enabled GRU and linear layers.

    Methods
    -------
    forward(input)
        Perform a forward pass through the GRU-based classifier.
    """

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
        batch_first=False,
        mask_init="uniform",
        mask_scale=1e-2,
        threshold_fn="binarizer",
        threshold=None,
        all_weights=[],
        seq_len=10,
        mask_weights=[],
    ):
        super(PBGRU, self).__init__()

        # Initialize parameters
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.all_weights = all_weights
        self.seq_len = seq_len
        self.mask_weights = mask_weights

        # Load weights
        self.gru_weight_ih_l0 = all_weights["gru.weight_ih_l0"]
        self.gru_weight_hh_l0 = all_weights["gru.weight_hh_l0"]
        self.gru_bias_ih_l0 = all_weights["gru.bias_ih_l0"]
        self.gru_bias_hh_l0 = all_weights["gru.bias_hh_l0"]
        self.linear_weight = all_weights["linear.weight"]
        self.linear_bias = all_weights["linear.bias"]
        self.GRU_weights = [
            self.gru_weight_ih_l0,
            self.gru_weight_hh_l0,
            self.gru_bias_ih_l0,
            self.gru_bias_hh_l0,
        ]
        self.linear_weights = [self.linear_weight, self.linear_bias]

        # Initialize mask weights
        if mask_weights != []:
            self.GRU_mask_weights = mask_weights[0:4]
            self.Linear_mask_weights = mask_weights[-1]
        else:
            self.GRU_mask_weights = []
            self.Linear_mask_weights = []

        # Define classifier
        self.classifier = nn.Sequential(
            nl.ElementWiseGRU(
                input_size=input_size,
                device=device,
                num_layers=num_layers,
                hidden_size=hidden_size,
                bias=bias,
                dropout=dropout,
                bidirectional=bidirectional,
                training=training,
                mask_init=mask_init,
                mask_scale=mask_scale,
                threshold_fn=threshold_fn,
                threshold=threshold,
                GRU_weights=self.GRU_weights,
                seq_len=self.seq_len,
                GRU_mask_weights=self.GRU_mask_weights,
            ),
            nl.ElementWiseLinear(
                in_features=hidden_size,
                out_features=output_size,
                mask_init=mask_init,
                mask_scale=mask_scale,
                threshold_fn=threshold_fn,
                threshold=threshold,
                linear_weights=self.linear_weights,
                Linear_mask_weights=self.Linear_mask_weights,
            ),
        )

    def forward(self, input):
        """
        Perform a forward pass through the GRU-based classifier.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor of shape (batch_size, seq_len, input_size).

        Returns
        -------
        out : torch.Tensor
            The output tensor of shape (batch_size, seq_len, output_size).
        """
        out = self.classifier(input)
        return out


class PBLSTM(nn.Module):
    """
    A piggyback-enabled LSTM (Long Short-Term Memory) model with element-wise masking
    for fine-grained weight adjustments.

    (Parameters, attributes, and methods are the same as cPBGRU but tailored for LSTM.)

    """

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
        batch_first=False,
        mask_init="uniform",
        mask_scale=1e-2,
        threshold_fn="binarizer",
        threshold=None,
        all_weights=[],
        seq_len=10,
        mask_weights=[],
    ):
        super(PBLSTM, self).__init__()

        # Initialize parameters
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.all_weights = all_weights
        self.seq_len = seq_len
        self.mask_weights = mask_weights

        # Load weights
        self.lstm_weight_ih_l0 = all_weights["lstm.weight_ih_l0"]
        self.lstm_weight_hh_l0 = all_weights["lstm.weight_hh_l0"]
        self.lstm_bias_ih_l0 = all_weights["lstm.bias_ih_l0"]
        self.lstm_bias_hh_l0 = all_weights["lstm.bias_hh_l0"]
        self.linear_weight = all_weights["linear.weight"]
        self.linear_bias = all_weights["linear.bias"]
        self.LSTM_weights = [
            self.lstm_weight_ih_l0,
            self.lstm_weight_hh_l0,
            self.lstm_bias_ih_l0,
            self.lstm_bias_hh_l0,
        ]
        self.linear_weights = [self.linear_weight, self.linear_bias]

        # Initialize mask weights
        if mask_weights != []:
            self.LSTM_mask_weights = mask_weights[0:4]
            self.Linear_mask_weights = mask_weights[-1]
        else:
            self.LSTM_mask_weights = []
            self.Linear_mask_weights = []

        # Define classifier
        self.classifier = nn.Sequential(
            nl.ElementWiseLSTM(
                input_size=input_size,
                device=device,
                num_layers=num_layers,
                hidden_size=hidden_size,
                bias=bias,
                dropout=dropout,
                bidirectional=bidirectional,
                training=training,
                mask_init=mask_init,
                mask_scale=mask_scale,
                threshold_fn=threshold_fn,
                threshold=threshold,
                LSTM_weights=self.LSTM_weights,
                seq_len=self.seq_len,
                LSTM_mask_weights=self.LSTM_mask_weights,
            ),
            nl.ElementWiseLinear(
                in_features=hidden_size,
                out_features=output_size,
                mask_init=mask_init,
                mask_scale=mask_scale,
                threshold_fn=threshold_fn,
                threshold=threshold,
                linear_weights=self.linear_weights,
                Linear_mask_weights=self.Linear_mask_weights,
            ),
        )

    def forward(self, input):
        """
        Perform a forward pass through the LSTM-based classifier.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor of shape (batch_size, seq_len, input_size).

        Returns
        -------
        out : torch.Tensor
            The output tensor of shape (batch_size, seq_len, output_size).
        """
        out = self.classifier(input)
        return out
