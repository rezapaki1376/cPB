from typing import Callable


class LearnerConfig:
    """
    Class that implements the configurations for the benchmark framework.
    """

    def __init__(
        self,
        name: str,
        model: Callable,
        numeric: bool = None,
        batch_learner: bool = True,
        drift: bool = None,
        cpnn: bool = False,
        smart: bool = False,
    ):
        """

        Parameters
        ----------
        name: str
            The name of the model.
        model: func
            The function that creates the model and returns it.
        numeric: bool, default: None.
            True if the model deals with numpy values, False if it deals with dict (e.g. Decision Tress).
            If None, it's set to True in case of batch learner or cpnn.
            It's automatically set to True in case of cpnn.
        batch_learner: bool, default: True.
            True if the model is a periodic learner, False otherwise.
        drift: bool, default: None.
            True if the model must handle drifts using the method add_new_column. False otherwise.
            If None is set to True in case of cpnn or batch_learner.
        cpnn: bool, default: False.
            True if the model is a cPNN based model, False otherwise.
        smart: bool, default: False.
            True if the model is a Smart cPNN. False otherwise.
        """
        self.name = name
        self.model = model
        self.cpnn = cpnn
        self.drift = drift
        self.batch_learner = batch_learner
        self.numeric = numeric
        self.smart = smart

        if self.cpnn:
            self.numeric = True
        else:
            self.smart = False
        if self.numeric is None:
            if self.batch_learner or self.cpnn:
                self.numeric = True
            else:
                self.numeric = False
        if self.drift is None:
            if self.cpnn:
                self.drift = True
            elif self.batch_learner:
                self.drift = True
            else:
                self.drift = False
