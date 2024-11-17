from river import stream
from models.federated_cpnn import *
from evaluation.prequential_evaluation import EvaluatePrequential, make_dir
import pandas as pd
import sys
import traceback
from evaluation.test_utils import *

# __________________
# PARAMETERS
# __________________
PATHS = [
    "datasets/weather_1conf_fcpnn"
]  # a list containing the paths of the data streams (without the extension)
SEQ_LEN = 11  # length of the sequence, 11 for Weather, 10 for Sine
ITERATIONS = 10  # number of experiments
PATH_PERFORMANCE = "fcpnn"  # path to write the outputs of the evaluation
CALLBACK_FUNC = None  # function to call after each iteration (set to callback_func_federated to write fcpnn selections)
MODE = "local"  # 'local' or 'aws'. If 'aws', the messages will be written in a specific txt file in the output_file dir
OUTPUT_FILE = None
# the name of the output file in outputs dir. If None, it will use the name of the current data stream.
BATCH_SIZE = 128  # the batch size of periodic learners and classifiers.


# __________________
# CODE
# __________________
NUM_FEATURES = 2
NUM_CLASSES = 2
NUM_OLD_LABELS = SEQ_LEN - 1
POSITION = 5000
WIDTH = 1
METRICS = ["accuracy", "kappa"]
MAX_SAMPLES = None
WRITE_CHECKPOINTS = False
INITIAL_TASK = 1
DF = pd.DataFrame
DRIFTS = []

if OUTPUT_FILE is None:
    OUTPUT_FILE = PATHS[0].split("/")[-1]

initialize(NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, ITERATIONS)
eval_cl = None


def get_drifts():
    prev_task = DF.iloc[0]["task"]
    drifts = []
    cont = 0
    for i, row in DF.iterrows():
        if prev_task != row["task"]:
            drifts.append(cont)
            prev_task = row["task"]
        cont += 1
    return drifts


def create_iter_pandas_12():
    X = DF.iloc[: DRIFTS[1]].drop(columns="target").values
    y = DF.iloc[: DRIFTS[1]]["target"].values
    feature_names = [f"x{i}" for i in range(1, X.shape[1])] + ["task"]
    return stream.iter_array(X, y, feature_names=feature_names)


def create_iter_pandas_23():
    X = DF.iloc[DRIFTS[0] : DRIFTS[2]].drop(columns="target").values
    y = DF.iloc[DRIFTS[0] : DRIFTS[2]]["target"].values
    feature_names = [f"x{i}" for i in range(1, X.shape[1])] + ["task"]
    return stream.iter_array(X, y, feature_names=feature_names)


def create_iter_pandas_final():
    X = DF.iloc[DRIFTS[2] :].drop(columns="target").values
    y = DF.iloc[DRIFTS[2] :]["target"].values
    feature_names = [f"x{i}" for i in range(1, X.shape[1])] + ["task"]
    return stream.iter_array(X, y, feature_names=feature_names)


def create_federated():
    model_name = list(eval_preq_12._eval.keys())[0]
    models = [
        eval_preq_12._eval[model_name]["alg"][0],
        eval_preq_23._eval[model_name]["alg"][0],
    ]
    federated = FederatedCPNN(models, num_batches=50)
    federated.add_new_column(DF.iloc[DRIFTS[2]]["task"])
    return federated


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
        initial_task_id=INITIAL_TASK,
    )


anytime_learners = []

batch_learners = [
    LearnerConfig(
        name="Q-cPNN",
        model=create_qcpnn_clstm,
        numeric=True,
        batch_learner=True,
        drift=True,
        cpnn=True,
    )
]

batch_learners_federated = [
    LearnerConfig(
        name="F-cPNN",
        model=create_federated,
        numeric=True,
        batch_learner=True,
        drift=True,
        cpnn=True,
    )
]

PATH = ""
if not PATH_PERFORMANCE.startswith("/"):
    PATH_PERFORMANCE = os.path.join("performance", PATH_PERFORMANCE)

orig_stdout = sys.stdout
f = None
if MODE == "aws":
    make_dir(f"outputs")
    f = open(f"outputs/{OUTPUT_FILE}.txt", "w", buffering=1)
    sys.stdout = f

try:
    for path in PATHS:
        PATH = path
        current_path_performance = os.path.join(PATH_PERFORMANCE, PATH.split("/")[-1])
        make_dir(current_path_performance)

        DF = pd.read_csv(f"{PATH}.csv")
        DRIFTS = get_drifts()
        columns = list(DF.columns)
        columns.remove("target")
        columns.remove("task")
        NUM_FEATURES = len(columns)

        initialize(NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, iterations_=1)
        print(PATH)
        print("BATCH SIZE, SEQ LEN:", BATCH_SIZE, SEQ_LEN)
        print("NUM OLD LABELS:", NUM_OLD_LABELS)
        print("ANYTIME LEARNERS:", [m.name for m in anytime_learners])
        print("BATCH LEARNERS:", [(m.name, m.drift) for m in batch_learners])
        print()

        for it in range(1, ITERATIONS + 1):
            print(f"ITERATION {it}/{ITERATIONS}")
            print("FIRST LOCAL MODEL")
            INITIAL_TASK = int(DF.iloc[0]["task"])
            eval_preq_12 = EvaluatePrequential(
                max_data_points=MAX_SAMPLES,
                batch_size=BATCH_SIZE,
                metrics=METRICS,
                anytime_learners=anytime_learners,
                batch_learners=batch_learners,
                data_stream=create_iter_pandas_12,
                path_write=current_path_performance,
                suffix=f"_local-12_it-{it}",
                write_checkpoints=WRITE_CHECKPOINTS,
                iterations=1,
                dataset_name=PATH.split("/")[-1],
                mode=MODE,
                anytime_scenario=False
            )
            initialize_callback(None, eval_preq_12)
            eval_preq_12.evaluate(callback=CALLBACK_FUNC, initial_task=INITIAL_TASK)
            model_name = list(eval_preq_12._eval.keys())[0]
            model_12: cPNN = eval_preq_12._eval[model_name]["alg"][0]
            if len(model_12.columns.columns) > 2:
                model_12.remove_last_column()

            print()
            print("SECOND LOCAL MODEL")
            INITIAL_TASK = int(DF.iloc[DRIFTS[0]]["task"])
            eval_preq_23 = EvaluatePrequential(
                max_data_points=MAX_SAMPLES,
                batch_size=BATCH_SIZE,
                metrics=METRICS,
                anytime_learners=anytime_learners,
                batch_learners=batch_learners,
                data_stream=create_iter_pandas_23,
                path_write=current_path_performance,
                suffix=f"_local-23_it-{it}",
                write_checkpoints=WRITE_CHECKPOINTS,
                iterations=1,
                dataset_name=PATH.split("/")[-1],
                mode=MODE,
                anytime_scenario=False
            )
            initialize_callback(None, eval_preq_23)
            eval_preq_23.evaluate(callback=CALLBACK_FUNC, initial_task=INITIAL_TASK)
            model_name =  list(eval_preq_23._eval.keys())[0]
            model_23: cPNN = eval_preq_23._eval[model_name]["alg"][0]
            if len(model_23.columns.columns) > 2:
                model_23.remove_last_column()

            print()
            print("FEDERATED MODEL")
            eval_preq_final = EvaluatePrequential(
                max_data_points=MAX_SAMPLES,
                batch_size=BATCH_SIZE,
                metrics=METRICS,
                anytime_learners=anytime_learners,
                batch_learners=batch_learners_federated,
                data_stream=create_iter_pandas_final,
                path_write=current_path_performance,
                suffix=f"_federated_it-{it}",
                write_checkpoints=WRITE_CHECKPOINTS,
                iterations=1,
                dataset_name=PATH.split("/")[-1],
                mode=MODE,
                anytime_scenario=False
            )
            initialize_callback(None, eval_preq_final)
            INITIAL_TASK = DF.iloc[DRIFTS[2]]["task"]
            eval_preq_final.evaluate(
                callback=callback_func_federated, initial_task=INITIAL_TASK
            )
            print("\n")
except Exception:
    print(traceback.format_exc())
    if MODE == "aws":
        sys.stdout = orig_stdout
        f.close()
        print(traceback.format_exc())
print("\n\nEND.")
if MODE == "aws":
    sys.stdout = orig_stdout
    f.close()
