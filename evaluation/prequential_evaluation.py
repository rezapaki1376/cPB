import pickle
import datetime
from typing import List, Iterable, Callable

from river import metrics
import os

# from metrics.kappa_t import CohenKappaTemporal
import numpy as np

from evaluation.learner_config import LearnerConfig
from Models.temporally_augmented_classifier import TemporallyAugmentedClassifier


def make_dir(path):
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)


# noinspection DuplicatedCode
class EvaluatePrequential:
    """
    Class that implements the comparison on a specific data stream between different models. See the method 'evaluate'
    for details
    """

    def __init__(
        self,
        max_data_points: int = None,
        batch_size: int = 128,
        metrics: Iterable = ("accuracy", "kappa"),
        anytime_learners: List[LearnerConfig] = None,
        batch_learners: List[LearnerConfig] = (),
        data_stream: Callable = None,
        path_write: str = None,
        train_test: bool = False,
        suffix: str = "",
        write_checkpoints: bool = False,
        iterations: int = 1,
        dataset_name: str = "",
        mode: str = "local",
        anytime_scenario: bool = True,
        periodic_scenario: bool = True,
        writing_frequency: int = 5000,
        drift_detector=None,
        iteration_str=None,
        initial_task=1,
    ):
        """
        Parameters
        ----------
        max_data_points: int, default: None
            If not None it indicates the number of data points to take from the data stream.
        batch_size: int, default: 128
            The batch size of the periodic learners.
        metrics: Iterable, default: ("accuracy", "kapp")
            The list of metrics to be computed. Available metrics: {'accuracy', 'kappa'}.
        anytime_learners: List[LearnerConfig], default: ()
            A list of LearnerConfig that contains the models able to learn from single data points.
            They must implement the method learn_one(x, y) and predict_one(x).
        batch_learners: List[LearnerConfig], default: ()
            A list of LearnerConfig that contains the periodic learners (models that learn from mini-batches).
            They must implement the method learn_many(x, y) and predict_many(x) and predict_one(x).
        data_stream: Callable, default: None.
            A function that returns the river iterator (e.g. iter_csv or iter_pandas) representing the data stream.
        path_write: str, default: None.
            The path to which write the evaluation outputs.
        train_test: bool, default: False.
            True if you want to add 'train_test' to the suffix of the evaluation outputs' file names.
        suffix: str, default: ''.
            The suffix to add the evaluation outputs' file names.
        write_checkpoints: bool, default: False.
            True if you want to write the pickles of the models after each task.
        iterations: int, default: 1.
            Number of experiments that you want to run.
        dataset_name: str, default: ''.
            The name of the dataset.
        mode: str, default: 'local'.
            'local' if you are running the experiment on your local machine.
            'aws' if you are running them on aws machines. In this case, it will write the messages in a specific file.
        anytime_scenario: bool, default: True.
            True if you want to perform test in the anytime classifier scenario (You can set both anytime and periodic
            scenarios to True).
        periodic_scenario: bool, default: True.
            True if you want to perform test in the periodic classifier scenario (You can set both anytime and periodic
            scenarios to True).
        writing_frequency: int, default: 5000.
            The frequency, specified as the number data points, at which the process writes the pickles files.
        drift_detector: detector, default: None.
            If provided, the automatic drift detector. The real drift are detected using the task labels of the
            data stream. This detector is only used to manage concept drift in the models. If None, it uses
            the task labels also to manage concept drift in the models.
        iteration_str: str, default: None.
            The string to append to each line when printing the outputs. If None, it prints the iteration index.
        initial_task: int, default: 1.
            The id of the first task. It should be equal to the first data point's task label.
        """

        super().__init__()
        self.max_data_points = max_data_points
        self.batch_size = batch_size
        self.metrics = metrics
        self._batch_learners_batch = list()
        self._anytime_learners_batch = list()
        self.anytime_learners = anytime_learners
        self.batch_learners = batch_learners
        self._write_checkpoints = write_checkpoints
        self._predictions = {}
        self._iterations = iterations
        self.dataset_name = dataset_name
        self.data_stream = data_stream
        self.path_write = path_write
        self.drift = False
        self.drift_detected = False
        self.suffix = "" if not train_test else "_train_test"
        self.suffix += suffix
        self.anytime_scenario = anytime_scenario
        self.periodic_scenario = periodic_scenario
        self._create_eval()
        self._mode = mode
        self.writing_freq = writing_frequency
        self.drift_detector = drift_detector
        self.detected_drifts = []
        self.iteration_str = iteration_str
        self.prev_task = initial_task
        make_dir(self.path_write)

    def _create_eval(self):
        self._eval = dict()
        self._perf = dict()
        self.checkpoint = dict()
        # Streaming Machine Learning models
        self._init_structure()

    def reset_checkpoints(self):
        for k in self.checkpoint:
            self.checkpoint[k] = [[] for i in range(0, self._iterations)]

    def _init_structure(self):
        models: List[LearnerConfig] = self.anytime_learners + self.batch_learners

        for model in models:
            name = model.name
            alg = model.model
            cpnn = model.numeric
            scenarios = []
            if self.periodic_scenario:
                scenarios.append("_batch")
            if self.anytime_scenario:
                scenarios.append("_anytime")
            for scenario in scenarios:
                self.checkpoint[name + scenario] = [
                    [] for i in range(0, self._iterations)
                ]
                self._perf[name + scenario] = {}
                self._eval[name + scenario] = {}
                if cpnn:
                    if scenario == "_batch":
                        self._eval[name + scenario]["alg"] = []
                        for i in range(self._iterations):
                            self._eval[name + scenario]["alg"].append(alg())
                    else:
                        if not self.periodic_scenario:
                            self._eval[name + "_batch"] = {}
                            self._eval[name + "_batch"]["alg"] = []
                            for i in range(self._iterations):
                                self._eval[name + "_batch"]["alg"].append(alg())
                        self._eval[name + "_anytime"]["alg"] = self._eval[
                            name + "_batch"
                        ]["alg"]
                else:
                    self._eval[name + scenario]["alg"] = []
                    for i in range(self._iterations):
                        self._eval[name + scenario]["alg"].append(alg())
                    if scenario == "_batch":
                        if "_TA" in name.upper():
                            for i in range(self._iterations):
                                self._eval[name + scenario]["alg"][
                                    i
                                ].set_use_predictions("both")
                self._eval[name + scenario]["metrics"] = {}
                self._perf[name + scenario]["drifts"] = [
                    [] for i in range(0, self._iterations)
                ]
                self._predictions[name + scenario] = [
                    [] for i in range(0, self._iterations)
                ]
                self._perf[name + scenario]["time"] = [
                    [] for i in range(0, self._iterations)
                ]
                if scenario == "_anytime":
                    metric_periods = ["all", "task"]
                else:
                    metric_periods = ["batch"]
                for t in metric_periods:
                    self._eval[name + scenario]["metrics"][t] = {}
                    self._perf[name + scenario][t] = {}
                    for metric in self.metrics:
                        self._perf[name + scenario][t][metric] = [
                            [] for i in range(0, self._iterations)
                        ]
                        self._eval[name + scenario]["metrics"][t][metric] = []
                        for i in range(self._iterations):
                            self._eval[name + scenario]["metrics"][t][metric].append(
                                self._get_metric(metric)
                            )

    @staticmethod
    def _get_metric(metric):
        if metric == "accuracy":
            return metrics.Accuracy()
        if metric == "kappa":
            return metrics.CohenKappa()
        # if metric == 'kappa_t':
        #     return CohenKappaTemporal()

    def _iter_anytime_learners(
        self, x, y, models: List[LearnerConfig], task=None, iteration=0
    ):
        i = 0
        for model in models:
            if model.numeric:
                x_ = list(x.values())
            else:
                x_ = x
            model_name = model.name + "_anytime"
            if self.drift_detected and model.drift:
                self._eval[model_name]["alg"][iteration].add_new_column(task)
            y_hat = self._eval[model_name]["alg"][iteration].predict_one(x_)
            self._predictions[model_name][iteration] += [y_hat]
            if self.drift:
                # reset 'task' metrics
                for metric in self.metrics:
                    self._eval[model_name]["metrics"]["task"][metric][
                        iteration
                    ] = self._get_metric(metric)
                self._perf[model_name]["drifts"][iteration].append(
                    len(self._perf[model_name]["all"][self.metrics[0]][iteration])
                )
                self.checkpoint[model_name][iteration].append(
                    pickle.loads(pickle.dumps(self._eval[model_name]["alg"][iteration]))
                )
            for metric in self.metrics:
                for t in ["all", "task"]:
                    # update the metrics
                    self._eval[model_name]["metrics"][t][metric][
                        iteration
                    ] = self._eval[model_name]["metrics"][t][metric][iteration].update(
                        y, y_hat
                    )
                    # get metrics
                    self._perf[model_name][t][metric][iteration].append(
                        self._eval[model_name]["metrics"][t][metric][iteration].get()
                    )
            start = datetime.datetime.now()
            self._eval[model_name]["alg"][iteration].learn_one(x_, y)
            end = datetime.datetime.now()
            self._perf[model_name]["time"][iteration].append((end - start).microseconds)
            i += 1

    def _iter_batch_learners_anytime_classification(self, x, y, idx, iteration=0):
        x = np.array([x[k] for k in x])
        for model in self.batch_learners:
            model_name = model.name
            model_name_pred = model_name + "_anytime"
            y_hat = self._eval[model_name + "_batch"]["alg"][iteration].predict_one(x)
            if self.drift:
                for metric in self.metrics:
                    self._eval[model_name_pred]["metrics"]["task"][metric][
                        iteration
                    ] = self._get_metric(metric)
                self._perf[model_name_pred]["drifts"][iteration].append(
                    len(self._perf[model_name_pred]["all"][self.metrics[0]][iteration])
                )
            self._predictions[model_name_pred][iteration] += [y_hat]
            for metric in self.metrics:
                for t in ["all", "task"]:
                    self._eval[model_name_pred]["metrics"][t][metric][
                        iteration
                    ] = self._eval[model_name_pred]["metrics"][t][metric][
                        iteration
                    ].update(
                        y, y_hat
                    )
                    self._perf[model_name_pred][t][metric][iteration].append(
                        self._eval[model_name_pred]["metrics"][t][metric][
                            iteration
                        ].get()
                    )

    def _iter_batch_models(
        self,
        x,
        y,
        models: List[LearnerConfig],
        add_checkpoint=True,
        task=None,
        iteration=0,
        add_new_column=True,
    ):
        # add new sample to the batch if there's no drift
        if len(models) == 0:
            return
        if models[0].batch_learner:
            batch = self._batch_learners_batch
            x = list(x.values())
        else:
            batch = self._anytime_learners_batch
        if not self.drift:
            batch.append((x, y))
        if len(batch) == self.batch_size or self.drift:
            x_batch, y_batch = list(zip(*batch))
            i = 0
            for m in models:
                model_name = m.name + "_batch"
                if (not m.cpnn or not m.batch_learner) or len(batch) >= self._eval[
                    model_name
                ]["alg"][iteration].get_seq_len():
                    if self.periodic_scenario:
                        if m.cpnn and m.batch_learner:
                            predictions = self._eval[model_name]["alg"][
                                iteration
                            ].predict_many(x_batch)
                        else:
                            predictions = []
                            for item in x_batch:
                                if m.numeric:
                                    item = list(item.values())
                                predictions.append(
                                    self._eval[model_name]["alg"][
                                        iteration
                                    ].predict_one(item)
                                )
                            predictions = np.array(predictions)
                        self._predictions[model_name][iteration] += list(predictions)
                        for y_true, y_hat in zip(y_batch, predictions):
                            for metric in self.metrics:
                                t = "batch"
                                self._eval[model_name]["metrics"][t][metric][
                                    iteration
                                ] = self._eval[model_name]["metrics"][t][metric][
                                    iteration
                                ].update(
                                    y_true, y_hat
                                )
                        for metric in self.metrics:
                            self._perf[model_name]["batch"][metric][iteration].append(
                                self._eval[model_name]["metrics"]["batch"][metric][
                                    iteration
                                ].get()
                            )
                    if m.cpnn and m.batch_learner:
                        start = datetime.datetime.now()
                        self._eval[model_name]["alg"][iteration].learn_many(
                            x_batch, y_batch
                        )
                        end = datetime.datetime.now()
                        if self.periodic_scenario:
                            self._perf[model_name]["time"][iteration].append(
                                (end - start).microseconds
                            )
                    else:
                        if self.periodic_scenario:
                            start = datetime.datetime.now()
                            for x_item, y_item in zip(x_batch, y_batch):
                                if m.numeric and type(x_item) == dict:
                                    x_item = list(x_item.values())
                                self._eval[model_name]["alg"][iteration].learn_one(
                                    x_item, y_item
                                )
                            end = datetime.datetime.now()
                            self._perf[model_name]["time"][iteration].append(
                                (end - start).microseconds
                            )
                elif (
                    m.cpnn
                    and m.batch_learner
                    and len(batch)
                    < self._eval[model_name]["alg"][iteration].get_seq_len()
                ):
                    if self.periodic_scenario:
                        self._predictions[model_name][iteration] += [None] * len(batch)
                        for metric in self.metrics:
                            self._perf[model_name]["batch"][metric][iteration] += [None]
                if self.periodic_scenario:
                    for metric in self.metrics:
                        self._eval[model_name]["metrics"]["batch"][metric][
                            iteration
                        ] = self._get_metric(metric)
                i += 1
            # reset the buffer for the next batch
            batch.clear()
            if self.drift:
                i = 0
                for m in models:
                    model_name = m.name + "_batch"
                    #if add_checkpoint:
                    #    if self.periodic_scenario:
                    #        self.checkpoint[model_name][iteration].append(
                    #            pickle.loads(
                    #                pickle.dumps(
                    #                    self._eval[model_name]["alg"][iteration]
                    #                )
                    #            )
                    #        )
                    if self.periodic_scenario:
                        self._perf[model_name]["drifts"][iteration].append(
                            len(
                                self._perf[model_name]["batch"][self.metrics[0]][
                                    iteration
                                ]
                            )
                        )
                    if m.cpnn and m.drift and add_new_column:
                        self._eval[model_name]["alg"][iteration].add_new_column(task)
                batch.append((x, y))
                i += 1

    def _write_pickles(self, iteration=0):
        if len(self.batch_learners) > 0:
            seq_len = f"_{self._eval[self.batch_learners[0].name+'_batch']['alg'][iteration].get_seq_len()}"
        elif len(self.anytime_learners) > 0:
            cpnn = False
            for i, m in enumerate(self.anytime_learners):
                if m.cpnn:
                    cpnn = True
                    break
            if cpnn:
                seq_len = f"_{self._eval[self.anytime_learners[i].name + '_anytime']['alg'][iteration].get_seq_len()}"
            else:
                seq_len = ""
                if len(self.anytime_learners) > 0:
                    if self.periodic_scenario:
                        for m in self.anytime_learners:
                            if "_TA" in m.name.upper():
                                seq_len = f"_{self._eval[m.name + '_batch']['alg'][iteration].num_old_labels+1}"
                            break
                    else:
                        for m in self.anytime_learners:
                            if "_TA" in m.name.upper():
                                seq_len = f"_{self._eval[m.name + '_anytime']['alg'][iteration].num_old_labels+1}"
                            break
        with open(
            os.path.join(
                self.path_write,
                f"performance_{self.batch_size}{seq_len}{self.suffix}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self._perf, f)
        with open(
            os.path.join(
                self.path_write,
                f"predictions_{self.batch_size}{seq_len}{self.suffix}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self._predictions, f)
        with open(
            os.path.join(
                self.path_write,
                f"drifts_{self.batch_size}{seq_len}{self.suffix}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.detected_drifts, f)
        if self._write_checkpoints:
            with open(
                os.path.join(
                    self.path_write,
                    f"checkpoint_{self.batch_size}{seq_len}{self.suffix}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(self.checkpoint, f)

    def evaluate(
        self,
        callback: Callable = None,
        initial_task: int = None,
        reset_checkpoints: bool = True,
        datastream: Callable = None,
        path_write=None,
        iteration_str=None,
    ):
        """
        It performs the prequential evaluation on anytime and periodic learners.
        All the models are run in both anytime and periodic classifier scenarios (if anytime_scenario and
        periodic_scenario are set to True).
        During the anytime classifier scenario, periodic learners learn on batches but predict on single data points.
        During the periodic classifier scenario, anytime learners learn and predict on batches. To do so, we
        iterate on the batch's data points and call the methods learn_one or predict_one.
        We instantiate different objects for the same anytime learner in different scenarios.

        Parameters
        ----------
        callback: Callable, default: None.
            A callback function to be called after each iteration. It will receive the following parameters:
            -   iteration: int. the number of the iteration (starting from 0)
            -   learners_dict: list. The list of LearnersConfig representing the periodic learners.
            -   path: str. The path_write parameter of the constructor.
            -   suffix: str. The suffix parameter of the constructor.
            -   models: dict. A dictionary containing for each name of the models in learners_dict the object
                representing the model (at the end of the iteration).
        initial_task: int, default: 1.
            The id of the first task. If you don't specify it and the first task id is not 1, it can raise an error.
        reset_checkpoints: bool, default: True.
            True if you want to reset the checkpoint after each iteration
        datastream: Callable, default: None.
            The function that creates the iterator on the da stream. If None, it uses the ones given in the
            constructor method.
        path_write: str, default: None.
            The path to which write the results. If None, it uses the ones given in the constructor method.
        iteration_str: str, default: None.
            The string to append when printing the outputs. If None, it uses the ones given in the constructor method.
        Returns
        -------
        It writes the following pickle files in the path_write/dataset_name path.
        -   performance_{batch_size}_{seq_len}{suffix}.pkl:
            It's a dict d with the structure: d['{name}_{scenario}'][key1]. Where:
            -   name is the name of the model specified in LearnerConfig (both anytime and periodic learners).
            -   scenario is the classifier scenario ('anytime' or 'batch' for periodic)

            If key1 is 'drifts', d['{name}_{scenario}']['time'][it] contains the list of the indexes of the first data
            points or batches (depending on the scenario) following a drift, during the iteration 'it'.

            If key1 is 'time', d['{name}_{scenario}']['time'][it] contains, for each data point or batch
            (depending on the scenario), the required time to perform the method learn_one or learn_many during the
            iteration 'it'.

            In the case of anytime classifier scenario, key1 could be also 'all' and 'task'.
            'all' indicates the metrics computed from the first data point of the data stream to the current one.
            'task' indicates the metrics that are reset after each drift.
            d['{name}_{scenario}'][key1][metric][it] will be a list containing the performance on each data point for
            the specific metric ('kappa' or 'accuracy') and the iteration 'it'.

            In the case of periodic classifier scenario, key1 could be also 'batch'.
            d['{name}_{scenario}']['batch'][metric][it] will be a list containing the performance on each batch for
            the specific metric ('kappa' or 'accuracy') and the iteration 'it'.
        - performance_{batch_size}_{seq_len}{suffix}.pkl:
            It's a dict d where d['{name}_{scenario}'][it] contains the list of the predictions made by the model
            with name 'name', in the scenario 'scenario', during the iteration 'it'.
        """
        if datastream is not None:
            self.data_stream = datastream
        if path_write is not None:
            self.path_write = path_write
        if iteration_str is not None:
            self.iteration_str = iteration_str
        for iteration in range(self._iterations):
            with open(os.path.join(self.path_write, "time.txt"), "a") as f:
                f.write(
                    f"START {iteration}/{self._iterations}: {datetime.datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}\n"
                )
            data_stream = self.data_stream()
            self._batch_learners_batch = list()
            self._anytime_learners_batch = list()
            if self._iterations == 1 and self.iteration_str is not None:
                iteration_str = self.iteration_str
            else:
                iteration_str = f"{iteration+1}/{self._iterations}"

            if self.max_data_points is None:
                stream = data_stream
            else:
                stream = data_stream.take(self.max_data_points)
            if initial_task is not None:
                self.prev_task = initial_task
            self.drift = False
            self.drift_detected = False
            drift_detector = pickle.loads(pickle.dumps(self.drift_detector))
            self.detected_drifts.append([])
            for idx, (x, y) in enumerate(stream):
                if (idx + 1) % 100 == 0:
                    print(
                        self.dataset_name,
                        f"{iteration_str}",
                        idx + 1,
                        end=None if self._mode == "aws" else "\r",
                    )

                supervised = False
                if "task" in x:
                    if self.prev_task != int(x["task"]):
                        self.drift = True
                    self.prev_task = int(x["task"])
                    del x["task"]
                    supervised = True
                if drift_detector is not None:
                    self.drift_detected = drift_detector.update(x, y)
                    if not supervised:
                        self.drift = self.drift_detected
                else:
                    self.drift_detected = self.drift

                if self.drift_detected:
                    self.detected_drifts[-1].append(idx)
                    print()
                    print("DETECTED DRIFT:", idx)
                elif self.drift:
                    print()
                    print("DRIFT:", idx)

                if self.anytime_scenario:
                    self._iter_anytime_learners(
                        x,
                        y,
                        self.anytime_learners,
                        task=self.prev_task,
                        iteration=iteration,
                    )
                    if not self.drift:
                        # if there is no drift, we must test the anytime model on the new data point before training on the
                        # batch
                        self._iter_batch_learners_anytime_classification(
                            x, y, idx, iteration=iteration
                        )
                if self.periodic_scenario:
                    self._iter_batch_models(
                        x,
                        y,
                        self.anytime_learners,
                        task=self.prev_task,
                        iteration=iteration,
                    )
                self._iter_batch_models(
                    x, y, self.batch_learners, task=self.prev_task, iteration=iteration
                )

                if self.drift and self.anytime_scenario:
                    # if there is a drift, we firstly must train the model on the last data points of previous concept,
                    # then we must add a new column, and finally we can test the anytime model on the new data point
                    self._iter_batch_learners_anytime_classification(
                        x, y, idx, iteration=iteration
                    )

                if self.drift:
                    self.drift = False

                if (idx + 1) % self.writing_freq == 0:
                    self._write_pickles(iteration)

            self.drift = True
            if self.periodic_scenario:
                self._iter_batch_models(
                    x,
                    y,
                    self.anytime_learners,
                    add_checkpoint=False,
                    iteration=iteration,
                    add_new_column=False,
                )
            self._iter_batch_models(
                x,
                y,
                self.batch_learners,
                add_checkpoint=False,
                iteration=iteration,
                add_new_column=False,
            )

            self._write_pickles(self._iterations - 1)

            if self.anytime_scenario:
                for m in self.anytime_learners:
                    self.checkpoint[m.name + "_anytime"][iteration].append(
                        pickle.loads(
                            pickle.dumps(
                                self._eval[m.name + "_anytime"]["alg"][iteration]
                            )
                        )
                    )

            if self.periodic_scenario:
                for m in self.anytime_learners + self.batch_learners:
                  	pass
                    #self.checkpoint[m.name + "_batch"][iteration].append(
                    #    pickle.loads(
                    #        pickle.dumps(
                    #            self._eval[m.name + "_batch"]["alg"][iteration]
                    #        )
                    #    )
                    #)

            if callback is not None:
                callback(
                    iteration=iteration,
                    learners_dict=self.anytime_learners,
                    path=self.path_write,
                    suffix=self.suffix,
                    models={
                        m.name: self._eval[m.name + "_anytime"]["alg"][iteration]
                        for m in self.anytime_learners
                    },
                )
            if reset_checkpoints:
                self.reset_checkpoints()
            print()
            with open(os.path.join(self.path_write, "time.txt"), "a") as f:
                f.write(
                    f"END {iteration}/{self._iterations}: {datetime.datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}\n"
                )
            print()
        return self._perf
