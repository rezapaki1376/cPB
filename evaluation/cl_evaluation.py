import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np
import math


class EvaluateContinualLearning:
    def __init__(
        self,
        path,
        checkpoint,
        anytime_learners,
        batch_learners,
        batch_size,
        path_write,
        train_test=False,
        suffix="",
    ):
        self.dataset = pd.read_csv(path + ".csv")
        self.anytime_learners = anytime_learners
        self.batch_learners = batch_learners
        self.X = []
        self.Y = []
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.feature_names = list(self.dataset.columns)[:-2]
        self._iterations = len(self.checkpoint[list(self.checkpoint.keys())[0]])
        for task in range(1, self.dataset["task"].max() + 1):
            df_task = self.dataset[self.dataset["task"] == task].drop(columns="task")
            self.X.append(df_task.iloc[:, :-1].values)
            self.Y.append(df_task.iloc[:, -1].values)
        for k in self.checkpoint:
            if "ens" in k.lower():
                del self.checkpoint[k]
        self.metric_names = ["kappa", "accuracy"]
        model_names = [b["name"] for b in self.batch_learners] + [
            a["name"] for a in self.anytime_learners
        ]
        self.metric_tables = {
            model: {
                metric: [[] for _ in range(self._iterations)]
                for metric in self.metric_names
            }
            for model in (
                [m + "_anytime" for m in model_names]
                + [m + "_batch" for m in model_names]
            )
        }
        self.cl_metrics = {}
        for model_name in model_names:
            for b in ["_batch", "_anytime"]:
                self.cl_metrics[model_name + b] = {}
                for metric in self.metric_names:
                    self.cl_metrics[model_name + b][metric] = [
                        [] for i in range(self._iterations)
                    ]
        self.path_write = path_write
        self.suffix = "" if not train_test else "_train_test"
        self.suffix += suffix

    def _compute_cl_metrics(self, model_name, metric, iteration=0):
        N = len(self.metric_tables[model_name][metric][iteration])
        self.cl_metrics[model_name][metric][iteration] = {
            "average": np.mean(self.metric_tables[model_name][metric][iteration]),
            "a_metric": np.sum(
                [
                    self.metric_tables[model_name][metric][iteration][i][j]
                    for i in range(N)
                    for j in range(i)
                ]
            )
            / (N * (N + 1) / 2),
            "bwt": np.mean(
                [
                    (
                        self.metric_tables[model_name][metric][iteration][-1][i]
                        - self.metric_tables[model_name][metric][iteration][i][i]
                    )
                    for i in range(N - 1)
                ]
            ),
        }

    def _convert_to_dict(self, x):
        return {self.feature_names[i]: x[i] for i in range(len(x))}

    def evaluate(self, iterations=None):
        if iterations is None:
            iterations = (0, self._iterations)
        for iteration in range(iterations[0], iterations[1]):
            for model_dict in self.anytime_learners:
                model_name = model_dict["name"] + "_anytime"
                for model in self.checkpoint[model_name][iteration]:
                    for metric_name in self.metric_names:
                        self.metric_tables[model_name][metric_name][iteration].append(
                            []
                        )
                    for task in range(len(self.X)):
                        if not (model_dict["numeric"]):
                            y_hat = [
                                model.predict_one(self._convert_to_dict(x))
                                for x in self.X[task]
                            ]
                        else:
                            y_hat = [model.predict_one(x) for x in self.X[task]]
                        y_true = self.Y[task]
                        assert len(y_hat) == len(y_true)
                        kappa = cohen_kappa_score(y_true, y_hat)
                        if math.isnan(kappa):
                            kappa = 1
                        self.metric_tables[model_name]["kappa"][iteration][-1].append(
                            kappa
                        )
                        self.metric_tables[model_name]["accuracy"][iteration][
                            -1
                        ].append(accuracy_score(y_true, y_hat))
                for metric in self.metric_names:
                    self.metric_tables[model_name][metric][iteration] = np.array(
                        self.metric_tables[model_name][metric][iteration]
                    )
                    self._compute_cl_metrics(model_name, metric, iteration)

            for model_dict in self.batch_learners:
                model_name_original = model_dict["name"]
                model_name = model_name_original + "_anytime"
                for model in self.checkpoint[model_name_original + "_batch"][iteration]:
                    for metric_name in self.metric_names:
                        self.metric_tables[model_name][metric_name][iteration].append(
                            []
                        )
                    for task in range(len(self.X)):
                        model.reset_previous_data_points()
                        y_hat = [
                            model.predict_one(
                                x, column_id=min(task, model.get_n_columns() - 1)
                            )
                            for x in self.X[task]
                        ]
                        y_true = [
                            self.Y[task][i]
                            for i in range(len(y_hat))
                            if y_hat[i] is not None
                        ]
                        y_hat = [y for y in y_hat if y is not None]
                        kappa = cohen_kappa_score(y_true, y_hat)
                        if math.isnan(kappa):
                            kappa = 1
                        self.metric_tables[model_name]["kappa"][iteration][-1].append(
                            kappa
                        )
                        self.metric_tables[model_name]["accuracy"][iteration][
                            -1
                        ].append(accuracy_score(y_true, y_hat))
                for metric in self.metric_names:
                    self.metric_tables[model_name][metric][iteration] = np.array(
                        self.metric_tables[model_name][metric][iteration]
                    )
                    self._compute_cl_metrics(model_name, metric, iteration)

            for cont_model, model_dict in enumerate(
                self.anytime_learners + self.batch_learners
            ):
                model_name = model_dict["name"] + "_batch"
                for model in self.checkpoint[model_name][iteration]:
                    for metric_name in self.metric_names:
                        self.metric_tables[model_name][metric_name][iteration].append(
                            []
                        )
                    batch_metrics = {k: [] for k in ["accuracy", "kappa"]}
                    for task in range(len(self.X)):
                        for i in range(0, len(self.X[task]), self.batch_size):
                            y_true = self.Y[task][i : i + self.batch_size]
                            if model_dict["batch"]:
                                y_hat = list(
                                    model.predict_many(
                                        self.X[task][i : i + self.batch_size],
                                        column_id=min(task, model.get_n_columns() - 1),
                                    )
                                )
                            else:
                                y_hat = []
                                for x in self.X[task][i : i + self.batch_size]:
                                    if not model_dict["numeric"]:
                                        y_hat += [
                                            model.predict_one(self._convert_to_dict(x))
                                        ]
                                    else:
                                        y_hat += [model.predict_one(x)]
                            assert len(y_hat) == len(y_true)
                            y_true = [
                                y_true[j]
                                for j in range(len(y_hat))
                                if y_hat[j] is not None
                            ]
                            y_hat = [y for y in y_hat if y is not None]
                            assert len(y_hat) == len(y_true)
                            if len(y_true) > 0:
                                kappa = cohen_kappa_score(y_true, y_hat)
                                if math.isnan(kappa):
                                    kappa = 1
                                batch_metrics["kappa"].append(kappa)
                                batch_metrics["accuracy"].append(
                                    accuracy_score(y_true, y_hat)
                                )
                        for metric in ["accuracy", "kappa"]:
                            self.metric_tables[model_name][metric][iteration][
                                -1
                            ].append(np.mean(batch_metrics[metric]))
                for metric in self.metric_names:
                    self.metric_tables[model_name][metric][iteration] = np.array(
                        self.metric_tables[model_name][metric][iteration]
                    )
                    self._compute_cl_metrics(model_name, metric, iteration)

            if len(self.batch_learners) > 0:
                model_isl = self.checkpoint[self.batch_learners[0]["name"] + "_batch"][
                    iteration
                ][0]
                seq_len = f"_{model_isl.get_seq_len()}"
            else:
                seq_len = ""
            with open(
                os.path.join(
                    self.path_write,
                    f"metric_tables_{self.batch_size}{seq_len}{self.suffix}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(self.metric_tables, f)
            with open(
                os.path.join(
                    self.path_write,
                    f"cl_metrics_{self.batch_size}{seq_len}{self.suffix}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(self.cl_metrics, f)
