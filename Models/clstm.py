import torch
from torch import nn
import numpy as np


class cLSTMLinear(nn.Module):
    def __init__(
        self,
        input_size=2,
        device=torch.device("cpu"),
        hidden_size=50,
        output_size=2,
        batch_size=128,
        many_to_one=False,
        remember_states = None
    ):
        super(cLSTMLinear, self).__init__()

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

    def forward(self, x, prev_h, train=False):
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
        s = torch.from_numpy(np.tile(state, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)