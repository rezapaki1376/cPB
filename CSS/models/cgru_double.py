import torch
from torch import nn
import numpy as np


class cGRULinearDouble(nn.Module):
    def __init__(
        self,
        input_size=2,
        device=torch.device("cpu"),
        hidden_size=128,
        output_size=2,
        batch_size=128,
        many_to_one=False,
        remember_states = False,
    ):
        super(cGRULinearDouble, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.h0 = np.zeros((1, self.hidden_size))
        self.many_to_one = many_to_one
        self.remember_states = remember_states
        self.many_to_one = many_to_one

        # LAYERS
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.gru1.to(self.device)
        #self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        #self.gru2.to(self.device)
        self.linear = nn.Linear(hidden_size, output_size)
        self.linear.to(self.device)
        
        
    def forward(self, x, prev_h, train=False):
        input_f = x.to(self.device)

        if prev_h is not None:
            input_f = torch.cat((x, prev_h), dim=2)  # (B, L, I+H)
        out_h1, _ = self.gru1(input_f, self._build_initial_state(x))
        #out_h2, _ = self.gru2(out_h1, self._build_initial_state(x))
        
        if self.many_to_one:
            out = self.linear(out_h1[:,-1,:])
        else:
            out = self.linear(out_h1)

        if train and self.remember_states:
            self.h0 = out_h1[:, 1, :].detach().numpy()

        return out, out_h1

        
    def _build_initial_state(self, x):
        s = torch.from_numpy(np.tile(self.h0, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)