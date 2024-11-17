from river import stream
from evaluation.cl_evaluation import EvaluateContinualLearning
from evaluation.learner_config import LearnerConfig
from evaluation.prequential_evaluation import EvaluatePrequential, make_dir
import pandas as pd
import sys
import traceback
from evaluation.test_utils import *

# __________________
# PARAMETERS
# __________________
PATHS = [
    "datasets/weather_1conf",
]  # a list containing the paths of the data streams (without the extension)
SEQ_LEN = 11  # length of the sequence, 11 for Weather, 10 for Sine
ITERATIONS = 10  # number of experiments
PATH_PERFORMANCE = "acpnn_qcpnn"  # path to write the outputs of the evaluation
CALLBACK_FUNC = None  # function to call after each iteration (set it to None)
MODE = "local"  # 'local' or 'aws'. If 'aws', the messages will be written in a specific txt file in the output_file dir
OUTPUT_FILE = None
# the name of the output file in outputs dir. If None, it will use the name of the current data stream.
suffix = f""  # the suffix to add the files containing the evaluation results.
BATCH_SIZE = 128  # the batch size of periodic learners and classifiers.

anytime_learners = anytime_learners_sml
# The list of LearnerConfig specifying the anytime learners.
# The associated models are able to perform training and inference on single data points.
# They must implement the methods learn_one(x, y), predict_one(x).
# Use anytime_learners_sml to add all the SML models. Use [] if you are not interested in testing anytime learners.
# Otherwise, specify a custom list of LearnerConfig.
batch_learners = batch_learners_acpnn_qcpnn
# The list of LearnerConfig specifying the periodic learners.
# The associated models are able to perform training only on mini-batches of data points.
# They must implement the methods learn_many(x, y), predict_many(x), predict_one(x).
# Use batch_learners_acpnn_qcpnn for the experiment on acpnn and qcpnn.
# Use batch_learners_fcpnn to run the single model for fcpnn experiment.
# Otherwise, specify a custom list of LearnerConfig.


# __________________
# CODE
# __________________
NUM_FEATURES = 2
NUM_CLASSES = 2
NUM_OLD_LABELS = SEQ_LEN - 1
METRICS = ["accuracy", "kappa"]
MAX_SAMPLES = None
TRAIN_TEST = False
WRITE_CHECKPOINTS = False
DO_CL = False
ANYTIME_SCENARIO = True
PERIODIC_SCENARIO = True

if OUTPUT_FILE is None:
    OUTPUT_FILE = PATHS[0].split("/")[-1]

initialize(NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, ITERATIONS)
eval_cl = None


def create_iter_csv():
    return stream.iter_csv(str(PATH) + ".csv", converters=converters, target="target")


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

        if TRAIN_TEST:
            PATH_CL = PATH + "_test"
            PATH = PATH + "_train"
        else:
            PATH_CL = PATH
        df = pd.read_csv(f"{PATH}.csv", nrows=1)
        columns = list(df.columns)
        initial_task = df.iloc[0]["task"]
        columns.remove("target")
        columns.remove("task")
        converters = {c: float for c in columns}
        converters["target"] = int
        converters["task"] = int
        NUM_FEATURES = len(columns)
        data_stream = create_iter_csv

        initialize(NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, ITERATIONS)
        print(PATH)
        print("BATCH SIZE, SEQ LEN:", BATCH_SIZE, SEQ_LEN)
        print("NUM OLD LABELS:", NUM_OLD_LABELS)
        print("TRAIN TEST:", TRAIN_TEST)
        print("ANYTIME LEARNERS:", [m.name for m in anytime_learners])
        print("BATCH LEARNERS:", [(m.name, m.drift) for m in batch_learners])
        print("SUFFIX:", suffix)
        print()

        eval_preq = EvaluatePrequential(
            max_data_points=MAX_SAMPLES,
            batch_size=BATCH_SIZE,
            metrics=METRICS,
            anytime_learners=anytime_learners,
            batch_learners=batch_learners,
            data_stream=data_stream,
            path_write=current_path_performance,
            train_test=TRAIN_TEST,
            suffix=suffix,
            write_checkpoints=WRITE_CHECKPOINTS,
            iterations=ITERATIONS,
            dataset_name=PATH.split("/")[-1],
            mode=MODE,
            anytime_scenario=ANYTIME_SCENARIO,
            periodic_scenario=PERIODIC_SCENARIO,
        )

        if PATH_CL is not None and DO_CL:
            eval_cl = EvaluateContinualLearning(
                path=PATH_CL,
                checkpoint=eval_preq.checkpoint,
                anytime_learners=anytime_learners,
                batch_learners=batch_learners,
                batch_size=BATCH_SIZE,
                path_write=current_path_performance,
                train_test=TRAIN_TEST,
                suffix=suffix,
            )

        initialize_callback(eval_cl, eval_preq)

        eval_preq.evaluate(callback=CALLBACK_FUNC, initial_task=initial_task)
        print()
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
