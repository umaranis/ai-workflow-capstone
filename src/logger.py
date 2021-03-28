import time
import os
import uuid
import csv
from datetime import date

if not os.path.exists(os.path.join(".", "logs")):
    os.mkdir("logs")

module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)
unittest_path = os.path.join(dir_path, "..", "test")

def update_train_log(data_shape, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test):
    """
    Update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    if test:
        logfile = os.path.join("logs", "train-test.log")
    else:
        logfile = os.path.join(unittest_path, "logs", "train-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file
    header = ['unique_id', 'timestamp', 'x_shape', 'model_version',
              'model_version_note', 'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), data_shape,
                             MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)


def update_predict_log(y_pred, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    if test:
        logfile = os.path.join("logs", "predict-test.log")
    else:
        logfile = os.path.join(unittest_path, "logs", "predict-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file
    header = ['unique_id', 'timestamp', 'y_pred', 'y_proba', 'query', 'model_version', 'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), y_pred,
                             MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)


if __name__ == "__main__":
    """
    basic test procedure for logger.py
    """

    from model import MODEL_VERSION, MODEL_VERSION_NOTE

    ## train logger
    update_train_log("121", "00:00:01",
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=False)
    ## predict logger
    update_predict_log("102", "00:00:01", MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
