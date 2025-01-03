import torch
from torch import nn
import torch.quantization
from Models.clstm import cLSTMLinear


class cPNNColumns(torch.nn.Module):
    """
    Class that implements the list of single cPNN columns.
    """

    def __init__(
        self,
        column_class=cLSTMLinear,
        device=None,
        lr=0.01,
        many_to_one=False,
        quantize=False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        column_class: default: cLSTMLinear
            The class that implements the single column's architecture.
        device: default: None.
            Torch's device, if None its value is set to 'cpu'.
        lr: float, default: 0.01.
            The learning rate value of columns' Adam Optimizer.
        many_to_one: bool, default: False
            If True each column is a many to one model.
        quantize: bool, default: False
            If True, after a concept drift, the column is quantized.
        kwargs:
            Parameters of column_class.
        """
        super(cPNNColumns, self).__init__()
        kwargs["device"] = (
            torch.device("cpu") if device is None else torch.device(device)
        )
        kwargs["many_to_one"] = many_to_one
        self.device = kwargs["device"]
        self.column_class = column_class
        self.column_args = kwargs
        self.lr = lr
        self.quantize = quantize

        self.columns = torch.nn.ModuleList([column_class(**kwargs)])
        self.column_args["input_size"] = (
            self.columns[0].input_size + self.columns[0].hidden_size
        )
        self.optimizers = [self._create_optimizer()]
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def _create_optimizer(self, column_id=-1):
        return torch.optim.Adam(self.columns[column_id].parameters(), lr=self.lr)

    def forward(self, x, column_id=None, train=False):
        if column_id is None:
            column_id = len(self.columns) - 1
        out = None
        prev_h = None
        for i in range(0, column_id + 1):
            out, prev_h = self.columns[i](x, prev_h, train=train)
        return out

    def forward_hidden(self, x, column_id=None):
        if column_id is None:
            column_id = len(self.columns) - 2
        out_h = None
        for i in range(0, column_id + 1):
            _, out_h = self.columns[i](x, out_h, train=False)
        return out_h

    def get_hidden(self, x, column_id=None):
        x = self.convert_to_tensor_dataset(x).to(self.device)

        if len(self.columns) == 1:
            return None

        if column_id is None:
            column_id = len(self.columns) - 2

        out_h = None
        for i in range(0, column_id + 1):
            _, out_h = self.columns[i](x, out_h)

        return out_h.detach().numpy()

    def add_new_column(self):
        """
        It adds a new column to the cPNN architecture, after a concept drift.
        Weights of previous columns are frozen.
        It also adds a new optimizer.
        """
        for param in self.columns[-1].parameters():
            param.requires_grad = False
        if self.quantize:
            last_column = self.columns[-1]
            self.columns = self.columns[:-1]
            conf = {}
            if "lstm" in self.column_class.__name__.lower():
                conf = {nn.LSTM, nn.Linear}
            if "gru" in self.column_class.__name__.lower():
                conf = {nn.GRU, nn.Linear}
            self.columns.append(
                torch.quantization.quantize_dynamic(
                    last_column, conf, dtype=torch.qint8
                )
            )
            self.columns[-1].quantized = True
        self.columns.append(self.column_class(**self.column_args))
        self.optimizers.append(self._create_optimizer())

    def unfreeze_last_column(self):
        if self.quantize:
            if self.columns[-1].quantized:
                last_column = self.columns[-1]
                self.columns = self.columns[:-1]
                column_args = self.column_args.copy()
                if len(self.columns) == 0:
                    column_args["input_size"] = last_column.input_size
                self.columns.append(self.column_class(**column_args))

                lstm_params = dict(last_column.lstm.get_weight())
                lstm_params.update(dict(last_column.lstm.get_bias()))
                for p in lstm_params:
                    lstm_params[p] = torch.dequantize(lstm_params[p])
                self.columns[-1].lstm.load_state_dict(lstm_params)

                linear_params = {
                    "weight": torch.dequantize(last_column.linear.weight()),
                    "bias": torch.dequantize(last_column.linear.bias()),
                }
                self.columns[-1].linear.load_state_dict(linear_params)
            self.columns[-1].quantized = False
        for param in self.columns[-1].parameters():
            param.requires_grad = True

    def remove_last_column(self):
        self.columns = self.columns[:-1]
        self.optimizers = self.optimizers[:-1]

    def take_first_columns(self, num_columns: int):
        self.columns = self.columns[:num_columns]
        self.optimizers = self.optimizers[:num_columns]
