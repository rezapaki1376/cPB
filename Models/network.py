import torch
import numpy as np
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from Models.PiggyBack import *
from Models.pretrain import *
import pickle


class ModifiedRNN(nn.Module):
    """
    A modified recurrent neural network (RNN) model supporting both GRU and LSTM architectures,
    with integrated piggyback masking for weights.

    Parameters
    ----------
    input_size : int, default=2
        The number of input features.
    device : torch.device, default=torch.device("cpu")
        The device to run the model on ('cpu' or 'cuda').
    num_layers : int, default=1
        The number of stacked RNN layers.
    hidden_size : int, default=50
        The number of features in the hidden state.
    output_size : int, default=2
        The number of output features.
    batch_size : int, default=128
        The number of samples per batch.
    base_model : str, default='gru'
        The base RNN type, either 'gru' or 'lstm'.
    many_to_one : bool, default=False
        Whether the model predicts only the last output (many-to-one).
    remember_states : bool or None, default=None
        Whether to remember and reuse hidden states across forward passes.
    bias : bool, default=True
        Whether to include a bias term in the RNN layers.
    dropout : float, default=0.0
        Dropout probability for RNN layers.
    training : bool, default=False
        Whether the model is in training mode.
    bidirectional : bool, default=False
        Whether the RNN layers are bidirectional.
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
    pretrain_model_addr : str, default=''
        Path to the pretrained model weights.
    seq_len : int, default=10
        The sequence length for input data.
    mask_weights : list, default=[]
        The weights used for masking in the piggyback model.

    Attributes
    ----------
    pretrain_model : nn.Module
        The pretrained RNN model (GRU or LSTM).
    classifier : nn.Module
        The piggyback model that incorporates the masking mechanism.
    all_weights : dict
        The state dictionary of the pretrained model.

    Methods
    -------
    forward(input)
        Perform a forward pass through the classifier.
    """

    def __init__(
        self,
        input_size=2,
        device=torch.device("cpu"),
        num_layers=1,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        base_model="GRU",
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
        pretrain_model_addr="",
        seq_len=10,
        mask_weights=[],
    ):
        """
        Initialize the ModifiedRNN model with support for pretrained GRU/LSTM models
        and piggyback masking.
        """
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
        self.pretrain_model_addr = pretrain_model_addr
        self.seq_len = seq_len
        self.mask_weights = mask_weights
        self.base_model = base_model

        # Load pretrained model (GRU or LSTM)
        if base_model == "GRU":
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

        # Load pretrained weights
        with open(self.pretrain_model_addr, "rb") as fp:
            self.pretrain_model.load_state_dict(pickle.load(fp), strict=False)
        self.all_weights = self.pretrain_model.state_dict()

        # Initialize the piggyback model
        if base_model == "GRU":
            self.classifier = PBGRU(
                input_size=input_size,
                device=device,
                num_layers=num_layers,
                hidden_size=hidden_size,
                output_size=output_size,
                batch_size=batch_size,
                many_to_one=many_to_one,
                remember_states=remember_states,
                bias=bias,
                training=training,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=batch_first,
                mask_init=mask_init,
                mask_scale=mask_scale,
                threshold_fn=threshold_fn,
                threshold=threshold,
                all_weights=self.all_weights,
                seq_len=seq_len,
                mask_weights=mask_weights,
            )
        else:
            self.classifier = PBLSTM(
                input_size=input_size,
                device=device,
                num_layers=num_layers,
                hidden_size=hidden_size,
                output_size=output_size,
                batch_size=batch_size,
                many_to_one=many_to_one,
                remember_states=remember_states,
                bias=bias,
                training=training,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=batch_first,
                mask_init=mask_init,
                mask_scale=mask_scale,
                threshold_fn=threshold_fn,
                threshold=threshold,
                all_weights=self.all_weights,
                seq_len=seq_len,
                mask_weights=mask_weights,
            )

    def forward(self, input):
        """
        Perform a forward pass through the classifier.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor of shape (batch_size, seq_len, input_size).

        Returns
        -------
        out : torch.Tensor
            The output tensor of shape (batch_size, seq_len, output_size)
            or (batch_size, 1, output_size) if many_to_one is True.
        """
        out = self.classifier(input)
        return out
