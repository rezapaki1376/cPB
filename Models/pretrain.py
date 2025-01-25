import torch
from torch import nn
import numpy as np


class GRU_Model(nn.Module):
    """
    A GRU-based model implemented using PyTorch's nn.Module.

    Parameters
    ----------
    input_size : int, default=2
        The number of features in the input data.
    device : torch.device, default=torch.device("cpu")
        The device to use for computations (e.g., 'cpu' or 'cuda').
    num_layers : int, default=1
        The number of stacked GRU layers.
    hidden_size : int, default=50
        The number of features in the hidden state of the GRU.
    output_size : int, default=2
        The number of output features.
    batch_size : int, default=128
        The number of samples per batch.
    many_to_one : bool, default=False
        Whether to use only the last output of the GRU (many-to-one).
    remember_states : bool or None, default=None
        Whether to remember the hidden states for subsequent forward passes.

    Attributes
    ----------
    gru : nn.GRU
        The GRU layer for processing sequential data.
    linear : nn.Linear
        The linear layer for mapping the GRU outputs to the desired output size.
    h0 : numpy.ndarray
        The initial hidden state for the GRU, initialized as zeros.
    device : torch.device
        The device to which the model and its tensors are moved.

    Methods
    -------
    forward(x, train=False)
        Perform a forward pass through the GRU model.
    _build_initial_state(x, state)
        Build the initial hidden state for the GRU.
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
    ):
        """
        Initialize the GRU model.

        Parameters
        ----------
        input_size : int
            Number of input features.
        device : torch.device
            Device to use for computation.
        num_layers : int
            Number of GRU layers.
        hidden_size : int
            Number of features in the GRU hidden state.
        output_size : int
            Number of output features.
        batch_size : int
            Batch size for input data.
        many_to_one : bool
            If True, only the last output is used for predictions.
        remember_states : bool or None
            If True, hidden states are remembered across forward passes.
        """
        super(GRU_Model, self).__init__()

         # PARAMETERS
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.h0 = torch.zeros(1, self.hidden_size, device=self.device)
        self.many_to_one = many_to_one
        self.remember_states = remember_states

        # LAYERS
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        self.to(self.device)

    def forward(self, x, prev_h=None, train=False):
        """
        Perform a forward pass through the GRU model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).
        train : bool, default=False
            If True and `remember_states` is enabled, hidden states are stored.

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape (batch_size, sequence_length, output_size)
            or (batch_size, 1, output_size) if many_to_one is True.
        """
        input_f = x.to(self.device)

        if prev_h is not None:
            prev_h = prev_h.to(self.device)
            input_f = torch.cat((x, prev_h), dim=2)  # (B, L, I+H)
        out_h, hn = self.gru(input_f, self._build_initial_state(x))
        if self.many_to_one:
            out = self.linear(out_h[:,-1,:])
        else:
            out = self.linear(out_h)

        # if we want to save first hidden state
        # if train and self.remember_states:
        #     self.h0 = out_h[:, 1, :].detach().numpy()
        # if we want to save last hidden state
        if train and self.remember_states:
            self.h0 = hn.detach().numpy()
            # or can be written like below
            # self.h0 = out_h[:, -1, :].detach().numpy()
        return out, out_h


class LSTM_Model(nn.Module):
    """
    An LSTM-based model implemented using PyTorch's nn.Module.

    Parameters
    ----------
    input_size : int, default=2
        The number of features in the input data.
    num_layers : int, default=1
        The number of stacked LSTM layers.
    device : torch.device, default=torch.device("cpu")
        The device to use for computations (e.g., 'cpu' or 'cuda').
    hidden_size : int, default=50
        The number of features in the hidden state of the LSTM.
    output_size : int, default=2
        The number of output features.
    batch_size : int, default=128
        The number of samples per batch.
    many_to_one : bool, default=False
        Whether to use only the last output of the LSTM (many-to-one).
    remember_states : bool or None, default=None
        Whether to remember the hidden states for subsequent forward passes.

    Attributes
    ----------
    lstm : nn.LSTM
        The LSTM layer for processing sequential data.
    linear : nn.Linear
        The linear layer for mapping the LSTM outputs to the desired output size.
    h0 : numpy.ndarray
        The initial hidden state for the LSTM, initialized as zeros.
    c0 : numpy.ndarray
        The initial cell state for the LSTM, initialized as zeros.
    device : torch.device
        The device to which the model and its tensors are moved.

    Methods
    -------
    forward(x, train=False)
        Perform a forward pass through the LSTM model.
    _build_initial_state(x, state)
        Build the initial hidden or cell state for the LSTM.
    """

    def __init__(
        self,
        input_size=2,
        num_layers=1,
        device=torch.device("cpu"),
        hidden_size=50,
        output_size=2,
        batch_size=128,
        many_to_one=False,
        remember_states=None,
    ):
        """
        Initialize the LSTM model.

        Parameters
        ----------
        input_size : int
            Number of input features.
        num_layers : int
            Number of LSTM layers.
        device : torch.device
            Device to use for computation.
        hidden_size : int
            Number of features in the LSTM hidden state.
        output_size : int
            Number of output features.
        batch_size : int
            Batch size for input data.
        many_to_one : bool
            If True, only the last output is used for predictions.
        remember_states : bool or None
            If True, hidden states are remembered across forward passes.
        """
        super(LSTM_Model, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.h0 = torch.zeros(1, self.hidden_size, device=self.device)
        self.c0 = torch.zeros(1, self.hidden_size, device=self.device)
        self.many_to_one = many_to_one
        self.remember_states = remember_states

        # LAYERS
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.to(self.device)

    def forward(self, x, prev_h=None, train=False):
        """
        Perform a forward pass through the LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).
        train : bool, default=False
            If True and `remember_states` is enabled, hidden states are stored.

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape (batch_size, sequence_length, output_size)
            or (batch_size, 1, output_size) if many_to_one is True.
        """
        input_f = x.to(self.device)

        if prev_h is not None:
            prev_h = prev_h.to(self.device)
            input_f = torch.cat((x, prev_h), dim=2)  # (B, L, I+H)

        out_h, (hn,cn) = self.lstm(
            input_f,
            (
                self._build_initial_state(x, self.h0),
                self._build_initial_state(x, self.c0),
            ),
        )
        if self.many_to_one:
            out = self.linear(out_h[:,-1:,:])
        else:
            out = self.linear(out_h)

        # if we want to save first hidden state
        # if train and self.remember_states:
        #     self.h0 = out_h[:, 1, :].detach().numpy()
        #     self.C0 = cn.detach().numpy()

        # if we want to save last hidden state
        if train and self.remember_states:
            self.h0 = hn.detach().numpy()
            # or can be written like below
            # self.h0 = out_h[:, 1, :].detach().numpy()
            self.C0 = cn.detach().numpy()

        return out, out_h

    def _build_initial_state(self, x, state):
        """
        Build the initial hidden or cell state for the LSTM.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).
        state : numpy.ndarray
            Initial state array (hidden or cell state).

        Returns
        -------
        s : torch.Tensor
            Initial state tensor of shape (num_layers, batch_size, hidden_size).
        """
        s = torch.from_numpy(np.tile(state, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)
