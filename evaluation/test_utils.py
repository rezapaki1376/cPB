import os
import pickle
import river
from river import tree
from evaluation.learner_config import LearnerConfig
from Models.clstm import cLSTMLinear
from Models.cpnn import cPNN
from Models.temporally_augmented_classifier import TemporallyAugmentedClassifier


NUM_OLD_LABELS = 0
SEQ_LEN = 0
NUM_FEATURES = 0
BATCH_SIZE = 0
ITERATIONS = 0
eval_cl = None
eval_preq = None


def initialize(num_old_labels_, seq_len_, num_features_, batch_size_, iterations_):
    global NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, ITERATIONS, eval_cl, eval_preq
    NUM_OLD_LABELS = num_old_labels_
    SEQ_LEN = seq_len_
    NUM_FEATURES = num_features_
    BATCH_SIZE = batch_size_
    ITERATIONS = iterations_


def initialize_callback(eval_cl_, eval_preq_):
    global eval_preq, eval_cl
    eval_cl = eval_cl_
    eval_preq = eval_preq_


def create_hat():
    return tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=100,
        split_confidence=1e-5,
        leaf_prediction="nb",
        nb_threshold=10,
    )


def create_hat_ta():
    return TemporallyAugmentedClassifier(
        base_learner=create_hat(),
        num_old_labels=NUM_OLD_LABELS,
    )


def create_arf():
    return river.ensemble.AdaptiveRandomForestClassifier(leaf_prediction="nb")


def create_arf_ta():
    return TemporallyAugmentedClassifier(
        base_learner=create_arf(),
        num_old_labels=NUM_OLD_LABELS,
    )


def create_cpnn_clstm():
    return cPNN(
        column_class=cLSTMLinear,
        device="cpu",
        seq_len=SEQ_LEN,
        train_verbose=False,
        input_size=NUM_FEATURES,
        output_size=2,
        hidden_size=50,
        batch_size=BATCH_SIZE,
    )


def create_qcpnn_clstm():
    return cPNN(
        column_class=cLSTMLinear,
        device="cpu",
        seq_len=SEQ_LEN,
        train_verbose=False,
        input_size=NUM_FEATURES,
        output_size=2,
        hidden_size=50,
        batch_size=BATCH_SIZE,
        qcpnn=True,
    )


def create_acpnn_clstm():
    return cPNN(
        column_class=cLSTMLinear,
        device="cpu",
        seq_len=SEQ_LEN,
        train_verbose=False,
        acpnn=True,
        input_size=NUM_FEATURES,
        output_size=2,
        hidden_size=50,
        batch_size=BATCH_SIZE,
    )


def callback_func_cl(**kwargs):
    if "iteration" in kwargs:
        iteration = kwargs["iteration"]
    else:
        iteration = None
    if iteration is None:
        iterations = (0, ITERATIONS)
    else:
        iterations = (iteration, iteration + 1)
    eval_cl.evaluate(iterations)


def callback_func_smart(**kwargs):
    if "iteration" in kwargs:
        iteration = kwargs["iteration"]
    else:
        iteration = None
    if iteration is None:
        iteration = 0
    selection = {}
    for model in kwargs["learners_dict"]:
        if model.smart:
            selection[model.name] = {
                "history": kwargs["models"][
                    model.name
                ].columns.selected_columns_history,
                "final": kwargs["models"][model.name].columns.final_selection,
            }
    with open(os.path.join(kwargs["path"], f"selections_{iteration+1}.pkl"), "wb") as f:
        pickle.dump(selection, f)


def callback_func_federated(**kwargs):
    if "suffix" in kwargs:
        iteration = kwargs["suffix"]
    else:
        iteration = None
    if iteration is None:
        iteration = 1
    selection = {}
    models = kwargs["models"]
    for model in models:
        selection[model] = {
            "columns_task_ids": [
                models[model].models[i].task_ids
                for i in range(len(models[model].models))
            ],
            "federated_task_dict": models[model].task_dict,
        }
    with open(os.path.join(kwargs["path"], f"task_ids_{iteration}.pkl"), "wb") as f:
        pickle.dump(selection, f)


anytime_learners_sml = [
    LearnerConfig(
        name="ARF",
        model=create_arf,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
    ),
    LearnerConfig(
        name="ARF_TA",
        model=create_arf_ta,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
    ),
    LearnerConfig(
        name="HAT",
        model=create_hat,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
    ),
    LearnerConfig(
        name="HAT_TA",
        model=create_hat_ta,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
    ),
]

anytime_learners_acpnn_qcpnn = anytime_learners_sml
batch_learners_acpnn_qcpnn = [
    LearnerConfig(
        name="cPNN",
        model=create_cpnn_clstm,
        numeric=True,
        batch_learner=True,
        drift=True,
        cpnn=True,
    ),
    LearnerConfig(
        name="Q-cPNN",
        model=create_qcpnn_clstm,
        numeric=True,
        batch_learner=True,
        drift=True,
        cpnn=True,
    ),
    LearnerConfig(
        name="A-cPNN",
        model=create_acpnn_clstm,
        numeric=True,
        batch_learner=True,
        drift=True,
        cpnn=True,
    ),
]

anytime_learners_fcpnn = []
batch_learners_fcpnn = [
    LearnerConfig(
        name="cPNN_one_model",
        model=create_cpnn_clstm,
        numeric=True,
        batch_learner=True,
        drift=True,
        cpnn=True,
    ),
]
